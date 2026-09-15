from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import sse
from .client import UpstreamError, stream_reply
from .config import ClientPool, load_config
from .context import (
    DEFAULT_IMAGE_INTENT_KEYWORDS,
    build_dialogue,
    extract_last_user_text,
    looks_like_image_request,
)
from .tools import build_tool_manifest, extract_tool_calls

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
    return {
        "object": "list",
        "data": [{"id": cfg.model.id, "object": "model", "owned_by": "yun139-fastapi"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, authorization: str | None = Header(None)):
    _check_auth(authorization)

    tool_manifest = build_tool_manifest(body.tools) if cfg.tools.enabled else None
    dialogue = build_dialogue(
        body.messages,
        tool_manifest,
        cfg.yun139.max_chars_per_request,
        cfg.yun139.oversized_context_strategy,
    )

    if cfg.yun139.enable_image_generation and cfg.yun139.image_intent_bypass:
        last_user_text = extract_last_user_text(body.messages)
        keywords = cfg.yun139.image_intent_keywords or DEFAULT_IMAGE_INTENT_KEYWORDS
        if last_user_text and looks_like_image_request(last_user_text, keywords):
            logger.info(
                "image-intent bypass triggered — sending raw last user message "
                "instead of the wrapped dialogue (was %d chars)", len(dialogue),
            )
            dialogue = last_user_text

    client_cfg = pool.next()
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
        try:
            async for delta, is_final in stream_reply(
                dialogue,
                client_cfg,
                cfg.yun139.upstream_url,
                cfg.yun139.idle_timeout_s,
                cfg.yun139.hard_timeout_s,
                cfg.yun139.enable_network_search,
                cfg.yun139.enable_image_generation,
            ):
                if delta:
                    full_text += delta
                    # While tool-calling is enabled we can't know a chunk is
                    # "safe" plain content until we see whether the full
                    # reply turns out to contain a tool_calls JSON block —
                    # so buffer and only stream deltas live when tools are
                    # off. With tools on, emit at the end instead.
                    if not cfg.tools.enabled:
                        yield sse.sse(sse.chunk(id_, created, model_id, {"content": delta}, None))
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
            yield sse.sse(sse.chunk(id_, created, model_id, {"content": full_text}, None))

        if tool_calls:
            yield sse.sse(
                sse.chunk(id_, created, model_id, {"tool_calls": tool_calls}, None)
            )
            yield sse.sse(sse.chunk(id_, created, model_id, {}, "tool_calls"))
        else:
            yield sse.sse(sse.chunk(id_, created, model_id, {}, "stop"))
        yield sse.sse_done()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
