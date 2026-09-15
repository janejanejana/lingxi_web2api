from __future__ import annotations

import json
import time
import uuid
from typing import Any


def new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def now_ts() -> int:
    return int(time.time())


def sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def sse_done() -> bytes:
    return b"data: [DONE]\n\n"


def chunk(id_: str, created: int, model: str, delta: dict, finish_reason: str | None) -> dict:
    return {
        "id": id_,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def completion(id_: str, created: int, model: str, content: str, tool_calls: list | None) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None
        finish_reason = "tool_calls"
    return {
        "id": id_,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
