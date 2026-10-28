from __future__ import annotations

import json
import re
import uuid
from typing import Any

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_TOOL_CALL_FENCE = "```json"
_TOOL_CALL_KEY = '\"tool_calls\"'


def _tool_call_prefix_status(candidate: str) -> str:
    remainder = candidate[len(_TOOL_CALL_FENCE):]
    position = 0
    while position < len(remainder) and remainder[position].isspace():
        position += 1
    if position == len(remainder):
        return "possible"
    if remainder[position] != "{":
        return "invalid"
    position += 1
    while position < len(remainder) and remainder[position].isspace():
        position += 1
    key_prefix = remainder[position:]
    if len(key_prefix) < len(_TOOL_CALL_KEY):
        return "possible" if _TOOL_CALL_KEY.startswith(key_prefix) else "invalid"
    return "match" if key_prefix.startswith(_TOOL_CALL_KEY) else "invalid"


class ToolCallStreamFilter:
    """Pass ordinary text through while withholding a likely tool-call block."""

    def __init__(self) -> None:
        self._candidate = ""
        self._withholding = False
        self._held = ""

    def feed(self, delta: str) -> str:
        if self._withholding:
            self._held += delta
            return ""

        text = self._candidate + delta
        self._candidate = ""
        output: list[str] = []

        while text:
            marker_at = text.find(_TOOL_CALL_FENCE)
            if marker_at < 0:
                marker_prefix_length = min(len(text), len(_TOOL_CALL_FENCE) - 1)
                while marker_prefix_length and not _TOOL_CALL_FENCE.startswith(text[-marker_prefix_length:]):
                    marker_prefix_length -= 1
                output.append(text[:-marker_prefix_length] if marker_prefix_length else text)
                self._candidate = text[-marker_prefix_length:] if marker_prefix_length else ""
                break

            output.append(text[:marker_at])
            self._candidate = _TOOL_CALL_FENCE
            text = text[marker_at + len(_TOOL_CALL_FENCE):]

            while text:
                self._candidate += text[0]
                text = text[1:]
                status = _tool_call_prefix_status(self._candidate)
                if status == "match":
                    self._withholding = True
                    self._held = self._candidate + text
                    self._candidate = ""
                    return "".join(output)
                if status == "invalid":
                    output.append(self._candidate)
                    self._candidate = ""
                    break

        return "".join(output)

    def finish(self, discard_tool_call: bool) -> str:
        if self._withholding:
            return "" if discard_tool_call else self._held
        return self._candidate


def build_tool_manifest(tools: list[dict] | None) -> str | None:
    """yun139's model has no native function-calling API — this is a
    best-effort text-protocol shim, same idea as what Gemini-FastAPI's own
    tool-calling support does on top of a web-only backend: describe the
    tools in plain language and ask for a specific JSON shape back."""
    if not tools:
        return None

    lines = [
        "System: You have access to the following tools. If, and only if, "
        "calling one of them is the right next step, reply with NOTHING "
        "except a single fenced code block in exactly this shape:\n"
        "```json\n"
        '{"tool_calls": [{"name": "<tool_name>", "arguments": { ... }}]}\n'
        "```\n"
        "If no tool call is needed, just answer normally in plain text — "
        "do not use the JSON block for anything else.",
        "Available tools:",
    ]
    for t in tools:
        fn = t.get("function", t)  # tolerate either {function: {...}} or flat
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        lines.append(f"- {name}: {desc}\n  parameters schema: {json.dumps(params, ensure_ascii=False)}")

    return "\n".join(lines)


def extract_tool_calls(reply_text: str) -> list[dict[str, Any]] | None:
    """Looks for the fenced ```json block described in build_tool_manifest.
    Returns a list of OpenAI-shaped tool_calls dicts, or None if the reply
    doesn't contain a (valid) one — callers should fall back to treating
    the whole reply as normal assistant content in that case.
    """
    match = _JSON_BLOCK_RE.search(reply_text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

    raw_calls = parsed.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        return None

    tool_calls = []
    for call in raw_calls:
        name = call.get("name")
        if not name:
            continue
        arguments = call.get("arguments", {})
        tool_calls.append(
            {
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )

    return tool_calls or None
