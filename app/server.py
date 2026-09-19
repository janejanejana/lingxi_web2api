from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import sse
from .attachments import AttachmentError, build_attachment_field, validate_attachment
from .client import UpstreamError, stream_reply
from .config import ClientPool, load_config
from .context import (
    DEFAULT_IMAGE_INTENT_KEYWORDS,
    build_dialogue,
    extract_attachment_from_messages,
    extract_last_user_text,
    looks_like_image_request,
)
from .tools import build_tool_manifest, extract_tool_calls
from .upload import UploadError, upload_attachment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yun139-fastapi")

app = FastAPI(title="yun139-fastapi")
cfg = load_config()
pool = ClientPool(cfg.yun139.clients)


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: object | None = None


def _check_auth(authorization: str | None):
    if not cfg.server.api_key:
        return
    expected = f"Bearer {cfg.server.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/v1/models")
async def list_models():
    # Hermes and similar OpenAI-compatible clients often inspect the model
    # metadata for a direct capability flag before deciding whether to
    # allow image-generation requests. Expose the capability explicitly,
    # independent of whether the model is configured for generation on this
    # deployment, so the UI can recognize that this backend can do it when
    # the feature is enabled.
    model_entry = {
        "id": cfg.model.id,
        "object": "model",
        "owned_by": "yun139-fastapi",
        "created": int(time.time()),
        "root": cfg.model.id,
        "parent": None,
        "permission": [],
        "capabilities": {
            "text_generation": True,
            "tool_calls": cfg.tools.enabled,
            "vision": True,
            "image_generation": True,
        },
        "supports_vision": True,
        "supports_image_generation": True,
    }
    return {"object": "list", "data": [model_entry]}


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, authorization: str | None = Header(None)):
    _check_auth(authorization)

    client_cfg = pool.next()

    last_user_text = extract_last_user_text(body.messages)
    image_keywords = cfg.yun139.image_intent_keywords or DEFAULT_IMAGE_INTENT_KEYWORDS
    force_raw_image_request = (
        cfg.yun139.enable_image_generation
        and cfg.yun139.image_intent_bypass
        and last_user_text
        and looks_like_image_request(last_user_text, image_keywords)
    )

    # The Hermes tool schema is exactly what nudges the backend model toward
    # tool-calling JSON instead of the native image-generation flow. For a
    # prompt that clearly asks to generate an image, we must suppress the
    # tool manifest entirely and send the raw user text instead.
    tool_manifest = build_tool_manifest(body.tools) if cfg.tools.enabled and not force_raw_image_request else None
    dialogue = last_user_text if force_raw_image_request else build_dialogue(
        body.messages,
        tool_manifest,
        cfg.yun139.max_chars_per_request,
        cfg.yun139.oversized_context_strategy,
    )

    attachment_field = None
    attachment_file_name = None
    if cfg.yun139.enable_attachments:
        extracted = extract_attachment_from_messages(body.messages)
        if extracted:
            file_bytes, filename, content_type = extracted
            logger.info(
                "attachment detected in request: name=%r content_type=%s size=%d bytes",
                filename, content_type, len(file_bytes),
            )
            try:
                validate_attachment(filename, len(file_bytes))
            except AttachmentError as e:
                logger.warning("attachment rejected by validation: %s", e)
                return JSONResponse(status_code=400, content={"error": {"message": str(e)}})
            try:
                uploaded = await upload_attachment(file_bytes, filename, content_type, client_cfg)
            except UploadError as e:
                logger.exception("attachment upload failed")
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": f"attachment upload failed: {e}"}},
                )
            logger.info(
                "attachment upload succeeded: fileId=%s parentFileId=%s",
                uploaded.file_id, uploaded.parent_file_id,
            )
            if cfg.yun139.post_upload_delay_s > 0:
                # The upload backend (personal-kd-njs.yun.139.com) and the
                # chat backend (ai.yun.139.com) appear to be separate
                # systems — sending the chat request immediately after
                # /hcy/file/complete returns has been observed to get
                # silently rejected (empty reply, finishReason:stop at
                # block index 0, no real generation at all), presumably
                # because the chat side hasn't caught up with the new file
                # yet. A real user has this gap for free (upload progress
                # bar -> reading it -> clicking send); we don't, so we
                # wait for it explicitly.
                logger.info(
                    "waiting %.1fs before the chat request for upload propagation",
                    cfg.yun139.post_upload_delay_s,
                )
                await asyncio.sleep(cfg.yun139.post_upload_delay_s)
            attachment_field = build_attachment_field([uploaded])
            attachment_file_name = uploaded.name
            # Attachment turns get the same minimal-dialogue treatment as
            # the image-generation bypass — the real app itself sends just
            # "描述图片" for these, not a wrapped System/tool/history blob,
            # and there's no evidence the vision path expects (or even
            # tolerates) that wrapping.
            dialogue = extract_last_user_text(body.messages) or "请描述并分析这个附件的内容。"
            logger.info(
                "attachment uploaded (fileId=%s) — sending minimal dialogue instead "
                "of the wrapped one", uploaded.file_id,
            )
        else:
            logger.debug("enable_attachments is on but no image_url found in the last user message")

    if (
        not attachment_field
        and cfg.yun139.enable_image_generation
        and cfg.yun139.image_intent_bypass
        and last_user_text
        and looks_like_image_request(last_user_text, image_keywords)
    ):
        logger.info(
            "image-intent bypass triggered — sending raw last user message "
            "instead of the wrapped dialogue (was %d chars)", len(dialogue),
        )
        dialogue = last_user_text

    model_id = body.model or cfg.model.id
    id_ = sse.new_id()
    created = sse.now_ts()
    t0 = time.monotonic()
    logger.info("[%s] request start — dialogue_len=%d stream=%s", id_, len(dialogue), body.stream)

    if not body.stream:
        full_text = ""
        try:
            async for delta, is_final in stream_reply(
                dialogue,
                client_cfg,
                cfg.yun139.upstream_url,
                cfg.yun139.idle_timeout_s,
                cfg.yun139.hard_timeout_s,
                cfg.yun139.enable_network_search,
                cfg.yun139.enable_image_generation,
                attachment_field,
                attachment_file_name,
            ):
                full_text += delta
                if is_final:
                    break
        except UpstreamError as e:
            logger.exception("[%s] upstream error after %.1fs", id_, time.monotonic() - t0)
            return JSONResponse(status_code=502, content={"error": {"message": str(e)}})

        logger.info("[%s] request done in %.1fs — reply_len=%d", id_, time.monotonic() - t0, len(full_text))
        tool_calls = extract_tool_calls(full_text) if cfg.tools.enabled else None
        visible_text = full_text if not tool_calls else ""
        return sse.completion(id_, created, model_id, visible_text, tool_calls)

    async def event_stream():
        full_text = ""
        # Announce the assistant role before any content — some
        # OpenAI-compatible clients rely on this to correctly initialize
        # the message object (same fix applied in the Node version).
        yield sse.sse(sse.chunk(id_, created, model_id, {"role": "assistant", "content": ""}, None))
        agen = stream_reply(
            dialogue,
            client_cfg,
            cfg.yun139.upstream_url,
            cfg.yun139.idle_timeout_s,
            cfg.yun139.hard_timeout_s,
            cfg.yun139.enable_network_search,
            cfg.yun139.enable_image_generation,
            attachment_field,
            attachment_file_name,
        )
        keepalive_s = cfg.yun139.keepalive_interval_s
        pending: asyncio.Task | None = None
        try:
            while True:
                if pending is None:
                    # Run __anext__() as a background task so a keepalive
                    # timeout never cancels it — cancelling it would kill
                    # the actual in-flight read from upstream, silently
                    # losing the real result. We only ever poll this same
                    # task across multiple keepalive intervals.
                    pending = asyncio.ensure_future(agen.__anext__())
                done, _ = await asyncio.wait({pending}, timeout=keepalive_s)
                if pending not in done:
                    # Nothing new from upstream in this window (e.g. the
                    # ~50-70s of dead air while an image is generating) —
                    # send a bare SSE comment line, ignored by any
                    # spec-compliant parser, purely to keep bytes flowing
                    # so the client's own read-timeout doesn't fire first.
                    logger.debug("[%s] keepalive (%ds idle)", id_, keepalive_s)
                    yield b": keepalive\n\n"
                    continue

                task, pending = pending, None
                try:
                    delta, is_final = task.result()
                except StopAsyncIteration:
                    break

                if delta:
                    full_text += delta
                    # While tool-calling is enabled we can't know a chunk is
                    # "safe" plain content until we see whether the full
                    # reply turns out to contain a tool_calls JSON block —
                    # so buffer and only stream deltas live when tools are
                    # off. With tools on, emit at the end instead.
                    if not cfg.tools.enabled:
                        yield sse.sse(
                            sse.chunk(
                                id_, created, model_id,
                                {"content": delta}, None,
                            )
                        )
                if is_final:
                    break
        except UpstreamError as e:
            logger.exception("[%s] upstream error after %.1fs", id_, time.monotonic() - t0)
            yield sse.sse({"error": {"message": str(e)}})
            yield sse.sse_done()
            return

        logger.info("[%s] request done in %.1fs — reply_len=%d", id_, time.monotonic() - t0, len(full_text))
        tool_calls = extract_tool_calls(full_text) if cfg.tools.enabled else None
        if cfg.tools.enabled and not tool_calls:
            # Nothing streamed live above (tools mode buffers) — flush the
            # whole reply now.
            yield sse.sse(
                sse.chunk(
                    id_, created, model_id,
                    {"content": full_text}, None,
                )
            )

        if tool_calls:
            yield sse.sse(
                sse.chunk(id_, created, model_id, {"tool_calls": tool_calls}, None)
            )
            yield sse.sse(sse.chunk(id_, created, model_id, {}, "tool_calls"))
        else:
            yield sse.sse(sse.chunk(id_, created, model_id, {}, "stop"))
        yield sse.sse_done()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
