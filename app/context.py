from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("yun139.context")


def _flatten_content(content: Any) -> str:
    """OpenAI content can be a plain string or a list of content parts
    (text/image_url/etc). We only forward text parts — yun139's endpoint
    has no documented multimodal input path."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(parts)
    return ""


DEFAULT_IMAGE_INTENT_KEYWORDS = [
    "生成图片", "生成一张图", "生成一幅", "生成配图", "生成插画",
    "画一张", "画个", "画一幅", "帮我画", "配一张图",
    "draw a picture", "draw an image", "generate an image",
    "generate a picture", "create an image", "text-to-image",
]


def extract_last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _flatten_content(m.get("content"))
    return ""


def looks_like_image_request(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in keywords)


ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool result",
}


def _render_tool_calls(tool_calls: list[dict]) -> str:
    parts = []
    for call in tool_calls:
        fn = call.get("function", {})
        name = fn.get("name", "unknown")
        args = fn.get("arguments", "{}")
        parts.append(f"{name}({args})")
    return "; ".join(parts)


def _render_message(m: dict) -> str | None:
    """Renders one messages[] entry to a text line, or None if there's
    nothing to show. Critically, this does NOT drop an assistant message
    just because content is null — an assistant turn that only made a
    tool call (content: null, tool_calls: [...]) is exactly the kind of
    turn that must survive into the reconstructed prompt, or the model
    loses track of what it already asked to be done and starts confidently
    claiming things are finished without actually re-issuing the tool
    call (see README "Tool calling caveats" / the Hermes debugging
    thread this was found in)."""
    role = m.get("role", "user")
    label = ROLE_LABELS.get(role, role.capitalize())
    text = _flatten_content(m.get("content"))
    tool_calls = m.get("tool_calls")

    if tool_calls:
        call_text = f"[Called tool(s): {_render_tool_calls(tool_calls)}]"
        return f"{label}: {text}\n{call_text}" if text else f"{label}: {call_text}"

    if role == "tool":
        tool_call_id = m.get("tool_call_id")
        suffix = f" (for call {tool_call_id})" if tool_call_id else ""
        return f"{label}{suffix}: {text}" if text else None

    return f"{label}: {text}" if text else None


def build_dialogue(
    messages: list[dict],
    tool_manifest: str | None,
    max_chars: int,
    strategy: str,
) -> str:
    """Turns the full messages[] history (system prompt, prior turns, tool
    schema description) into ONE self-contained text block, since yun139's
    endpoint has no native multi-turn session we can safely reuse across
    stateless API calls. This is the actual fix for "generic answers" —
    the model now sees everything Hermes gave it, not just the last line.
    """
    turns = [line for m in messages if (line := _render_message(m)) is not None]

    if tool_manifest:
        turns.insert(0, tool_manifest)

    full = "\n\n".join(turns)
    full += (
        "\n\n--------------\n"
        "Continue the conversation above as the Assistant. Reply with your "
        "next message only."
    )

    if len(full) <= max_chars or strategy == "none":
        return full

    logger.warning(
        "context %d chars exceeds max_chars_per_request=%d — truncating oldest turns "
        "(strategy=%s). If this fires often, raise max_chars_per_request in config.yaml.",
        len(full), max_chars, strategy,
    )

    if strategy == "truncate":
        return _truncate(messages, tool_manifest, max_chars)

    # Unknown strategy — fail safe to truncate rather than send something
    # that might get silently rejected/cut off by upstream.
    return _truncate(messages, tool_manifest, max_chars)


def _truncate(messages: list[dict], tool_manifest: str | None, max_chars: int) -> str:
    """Keeps the system prompt (if any) and the most recent turns verbatim,
    drops the oldest turns first until it fits. This is a crude stand-in
    for Gemini-FastAPI's "compaction" strategy — real compaction would
    summarize the dropped turns with another model call instead of just
    cutting them; wire that in here if you need it."""
    system_turns = []
    other_turns = []
    for m in messages:
        line = _render_message(m)
        if line is None:
            continue
        (system_turns if m.get("role") == "system" else other_turns).append(line)

    header = []
    if tool_manifest:
        header.append(tool_manifest)
    header.extend(system_turns)

    budget = max_chars - sum(len(h) + 2 for h in header) - 200  # slack for footer
    kept: list[str] = []
    for line in reversed(other_turns):
        if budget - len(line) - 2 < 0:
            break
        kept.insert(0, line)
        budget -= len(line) + 2

    dropped = len(other_turns) - len(kept)
    if dropped:
        logger.warning(
            "dropped %d oldest turn(s) to fit max_chars_per_request budget "
            "(kept %d of %d)", dropped, len(kept), len(other_turns),
        )
    body = header + ([f"[... {dropped} earlier turn(s) omitted for length ...]"] if dropped else []) + kept
    full = "\n\n".join(body)
    full += (
        "\n\n--------------\n"
        "Continue the conversation above as the Assistant. Reply with your "
        "next message only."
    )
    return full
