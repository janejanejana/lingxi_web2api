from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

import httpx

from .attachments import ATTACHMENT_COMMAND
from .config import ClientCfg

logger = logging.getLogger("yun139")

BEIJING_TZ = timezone(timedelta(hours=8))


def _beijing_iso_now() -> str:
    return datetime.now(BEIJING_TZ).strftime("%Y-%m-%dT%H:%M:%S+08:00")


def _build_payload(
    dialogue: str,
    enable_network_search: bool,
    enable_image_generation: bool,
    attachment: dict | None = None,
    attachment_file_name: str | None = None,
) -> dict:
    ext_info = {"pcVersion": "1.7.0", "h5Version": "3.2.0", "dialogueType": "noAiEditDialogue"}
    if attachment_file_name:
        # Confirmed present alongside a populated attachment field in a
        # real capture — unclear if it matters for multi-file attachments
        # (only single-file has been captured), so we just use the first/
        # only file's name for now.
        ext_info["fileName"] = attachment_file_name

    return {
        "applicationType": "chat",
        "sessionId": "",
        "dialogueInput": {
            "dialogue": dialogue,
            "prompt": "",
            "inputTime": _beijing_iso_now(),
            # Confirmed from a real capture: non-null
            # {"command":"036","subCommand":"036006"} accompanies a
            # populated attachment — presumably signals "this turn
            # includes an attachment" to the backend.
            "command": ATTACHMENT_COMMAND if attachment else None,
            "resourceType": "0",
            "resourceId": "",
            "dialogueType": "0",
            "commandType": 1,
            "enableForceNetworkSearch": enable_network_search,
            "enableAllNetworkSearch": False,
            "enableAiSearch": False,
            "extInfo": json.dumps(ext_info),
            "versionInfo": {"pcVersion": "1.7.0", "h5Version": "3.2.0"},
            # enableLlmDescribe is the switch that makes the assistant
            # rewrite the user's request into an image-gen prompt and
            # actually call the Seedream text-to-image model when it
            # judges an image is wanted — observed via app-side capture,
            # off by default here since it adds real latency (~10s) and
            # returns 24h-expiring signed URLs that must be consumed
            # promptly.
            "toolSetting": {"imageToolSetting": {"enableLlmDescribe": enable_image_generation}},
            "attachment": attachment or {},
            "aiWritingSetting": {},
            "enableModelThinking": False,
            "enableKnowledgeAndNetworkSearch": False,
        },
        "sourceChannel": None,  # filled in by caller
        "userId": None,  # filled in by caller
        "continuationInfo": None,
    }


def _headers(client_cfg: ClientCfg) -> dict:
    return {
        "Host": "ai.yun.139.com",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"
        ),
        "sec-ch-ua-platform": '"Windows"',
        "Authorization": client_cfg.authorization,
        "x-yun-client-info": "4g||30|||||||1536/864|zh-CN||||",
        "sec-ch-ua": '"Chromium";v="152", "Not?A_Brand";v="24", "Google Chrome";v="152"',
        "sec-ch-ua-mobile": "?0",
        "x-yun-api-version": "v1",
        "x-yun-app-channel": client_cfg.source_channel,
        "accept": "text/event-stream",
        "DNT": "1",
        "Content-Type": "application/json",
        "x-yun-tid": str(uuid.uuid4()),
        "Origin": "https://appmail.mail.10086.cn",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Referer": "https://appmail.mail.10086.cn/",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7",
    }


def _parse_sse_block(block: str) -> dict | None:
    for line in block.split("\n"):
        if line.startswith("data:"):
            raw = line[len("data:"):].strip()
            if not raw or raw == "[DONE]":
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
    return None


class UpstreamError(RuntimeError):
    pass


def _render_placeholder_image_state(flow: dict) -> str:
    """The first resultType-4 block is a placeholder that says the image is
    still being generated. Emitting a visible text chunk here keeps the SSE
    stream alive for clients that timeout on long silence, while still
    waiting for the real file-bearing block to arrive."""
    if flow.get("resultType") != 4:
        return ""
    file_list = flow.get("fileList") or []
    if file_list:
        return ""
    return "\n\n正在为你生成图片，请稍候…"


def _render_image_result(flow: dict) -> str:
    """resultType 4 = image-generation result (only appears when
    enableLlmDescribe is on and the model decided to generate an image).
    The first resultType-4 block is just a "still generating" placeholder
    with no fileList; the final one carries fileList with signed URLs
    (valid ~24h per the observed X-Amz-Expires=86400) — render those as
    markdown images appended to the reply text."""
    if flow.get("resultType") != 4:
        return ""
    file_list = flow.get("fileList") or []
    if not file_list:
        return ""
    lines = ["\n"]
    for f in file_list:
        name = f.get("name", "image")
        url = f.get("content", "")
        if url:
            lines.append(f"![{name}]({url})")
    return "\n".join(lines) if len(lines) > 1 else ""


def _is_probable_image_generation_stream(flow: dict, enable_image_generation: bool) -> bool:
    """Image generation can legally produce a text block first, then a
    resultType-4 image block later. In that case a plain 'stop' on the
    text block is not terminal yet, so we must not quit the stream until
    either the image block arrives or the upstream goes quiet enough to
    trigger the idle timeout."""
    if not enable_image_generation:
        return False
    # The upstream uses resultType=1 for ordinary streamed text, including
    # the text preamble emitted before the later resultType=4 image block.
    return flow.get("resultType") in (0, 1, None)


async def stream_reply(
    dialogue: str,
    client_cfg: ClientCfg,
    upstream_url: str,
    idle_timeout_s: int,
    hard_timeout_s: int,
    enable_network_search: bool = False,
    enable_image_generation: bool = False,
    attachment: dict | None = None,
    attachment_file_name: str | None = None,
) -> AsyncIterator[tuple[str, bool]]:
    """Yields (delta_text, is_final) pairs. is_final=True marks the chunk
    that carries (or coincides with) the terminal finishReason — same
    three-layer detection worked out for the Node version: in-stream
    field check, idle watchdog, and a leftover-buffer flush at the end.
    """
    payload = _build_payload(
        dialogue, enable_network_search, enable_image_generation,
        attachment, attachment_file_name,
    )
    payload["sourceChannel"] = client_cfg.source_channel
    payload["userId"] = client_cfg.user_id

    deadline = time.monotonic() + hard_timeout_s
    buffer = ""

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream(
                "POST", upstream_url, headers=_headers(client_cfg), json=payload
            ) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    raise UpstreamError(f"upstream returned {resp.status_code}: {text[:500]!r}")

                last_byte_at = time.monotonic()
                start = last_byte_at
                total_chunks = 0
                total_bytes = 0
                aiter = resp.aiter_text()
                while True:
                    remaining_idle = idle_timeout_s - (time.monotonic() - last_byte_at)
                    remaining_hard = deadline - time.monotonic()
                    if remaining_hard <= 0:
                        logger.warning(
                            "hard timeout hit — finalizing (received %d chunks / %d bytes over %.1fs, "
                            "never saw a terminal finishReason)",
                            total_chunks, total_bytes, time.monotonic() - start,
                        )
                        break
                    timeout = max(0.1, min(remaining_idle, remaining_hard))
                    try:
                        chunk = await asyncio.wait_for(aiter.__anext__(), timeout=timeout)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        logger.warning(
                            "idle timeout hit — finalizing (no bytes for %ds, had received %d "
                            "chunks / %d bytes over %.1fs total)",
                            idle_timeout_s, total_chunks, total_bytes, time.monotonic() - start,
                        )
                        break

                    total_chunks += 1
                    total_bytes += len(chunk)
                    last_byte_at = time.monotonic()
                    buffer += chunk
                    blocks = buffer.split("\n\n")
                    buffer = blocks.pop()

                    for block in blocks:
                        obj = _parse_sse_block(block)
                        if not obj:
                            logger.debug(
                                "[t+%.1fs] non-data block (comment/keepalive/unparseable): %r",
                                time.monotonic() - start, block[:200],
                            )
                            continue
                        data = obj.get("data") or {}
                        flow = data.get("flowResult") or {}
                        top_finish = data.get("finishReason")
                        # Prefer the OUTER finishReason as the authoritative
                        # "whole response is done" signal. flowResult's own
                        # finishReason only marks one sub-segment as done —
                        # with image generation on, the text segment reports
                        # finishReason:"stop" while the outer one is still
                        # "processing" because an image-gen segment follows;
                        # keying off flowResult here would cut the stream
                        # before the image ever arrives.
                        finish_reason = top_finish or flow.get("finishReason")
                        delta = (flow.get("outContent") or "") + _render_placeholder_image_state(flow) + _render_image_result(flow)
                        is_image_result = flow.get("resultType") == 4 and bool((flow.get("fileList") or []))
                        is_placeholder_image = flow.get("resultType") == 4 and not (flow.get("fileList") or [])
                        is_final = bool(finish_reason and finish_reason != "processing")
                        if enable_image_generation and not is_image_result and _is_probable_image_generation_stream(flow, enable_image_generation):
                            # The real image-generation flow can emit a placeholder resultType-4
                            # block with no fileList first, then the actual URL-bearing image later.
                            # Treat the placeholder as non-terminal and only close when the actual
                            # file-bearing result arrives.
                            is_final = False
                        if enable_image_generation and is_placeholder_image:
                            is_final = False
                        logger.info(
                            "[t+%.1fs] block index=%s finishReason=%s resultType=%s outContent=%r image_result=%s placeholder=%s final=%s",
                            time.monotonic() - start, flow.get("index"), finish_reason,
                            flow.get("resultType"), delta, is_image_result, is_placeholder_image, is_final,
                        )
                        if delta or is_final:
                            yield delta, is_final
                        if is_final and not (enable_image_generation and not is_image_result):
                            return

                # Loop ended (idle/hard timeout, or upstream closed) without
                # ever seeing a terminal finishReason in-stream — try the
                # leftover buffer once before giving up on it.
                obj = _parse_sse_block(buffer) if buffer.strip() else None
                if obj:
                    data = obj.get("data") or {}
                    flow = data.get("flowResult") or {}
                    delta = (flow.get("outContent") or "") + _render_image_result(flow)
                    if delta:
                        yield delta, False
                yield "", True
        except httpx.HTTPError as e:
            raise UpstreamError(f"transport error: {e}") from e
