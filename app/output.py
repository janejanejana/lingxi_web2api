from __future__ import annotations


_REMOVE_TEXT = "AI助手不支持直接完成该操作。"
_REPLACEMENTS = (("中国移动", "CtrlAll.com"),)
_PATTERNS = (_REMOVE_TEXT, *(source for source, _ in _REPLACEMENTS))
_MAX_PATTERN_LENGTH = max(map(len, _PATTERNS))


def sanitize_reply(text: str) -> str:
    text = text.replace(_REMOVE_TEXT, "")
    for source, replacement in _REPLACEMENTS:
        text = text.replace(source, replacement)
    return text


class ReplyTextFilter:
    """Apply reply text replacements without breaking across stream chunks."""

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, text: str) -> str:
        self._pending += text
        keep_length = 0
        for length in range(1, min(len(self._pending), _MAX_PATTERN_LENGTH - 1) + 1):
            suffix = self._pending[-length:]
            if any(pattern.startswith(suffix) for pattern in _PATTERNS):
                keep_length = length
        if keep_length:
            safe_text = self._pending[:-keep_length]
            self._pending = self._pending[-keep_length:]
        else:
            safe_text, self._pending = self._pending, ""
        return sanitize_reply(safe_text)

    def finish(self) -> str:
        remaining = sanitize_reply(self._pending)
        self._pending = ""
        return remaining