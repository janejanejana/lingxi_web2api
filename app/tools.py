from __future__ import annotations

import ast
import concurrent.futures
import json
import os
import re
import subprocess
import uuid
from typing import Any

_FENCE_TRIGGERS = ("```", "[called tool", "[called tool", "[CALLED TOOL")
_CALLED_HEADER_RE = re.compile(r"\[(?:CALLED TOOL|Called tool\(s\)):\s*([a-zA-Z0-9_]+)\s*\(", re.IGNORECASE)


def _extract_balanced_blocks(text: str, open_char: str = "{", close_char: str = "}") -> list[str]:
    """按括号层级配对提取完整结构，彻底解决嵌套大括号被非贪婪正则截断的问题。"""
    blocks: list[str] = []
    depth = 0
    start = -1
    in_string = False
    quote_char = ""
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_string:
            if ch == quote_char:
                in_string = False
            continue
        if ch in ('"', "'"):
            in_string = True
            quote_char = ch
            continue
        if ch == open_char:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_char and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                blocks.append(text[start : i + 1])
                start = -1
    return blocks


def _parse_args_safely(raw_str: str) -> dict[str, Any]:
    """容错解析 JSON 或 Python 风格字典（兼容单引号、True/False/None）。"""
    raw_str = raw_str.strip()
    if not raw_str:
        return {}
    try:
        parsed = json.loads(raw_str)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        parsed_eval = ast.literal_eval(raw_str)
        if isinstance(parsed_eval, dict):
            return parsed_eval
        if isinstance(parsed_eval, tuple) and parsed_eval and isinstance(parsed_eval[0], dict):
            return parsed_eval[0]
    except Exception:
        pass

    return {"raw_args": raw_str}


def _format_openai_tool_calls(raw_calls: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    tool_calls = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if not name:
            continue
        arguments = call.get("arguments", {})
        tool_calls.append(
            {
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": json.dumps(arguments, ensure_ascii=False)
                    if isinstance(arguments, dict)
                    else str(arguments),
                },
            }
        )
    return tool_calls or None


class ToolCallStreamFilter:
    """流式输出前瞻拦截器：优化后对普通文本零延迟实时放行，仅在检测到工具特征时拦截。"""

    def __init__(self) -> None:
        self._buffer = ""
        self._withholding = False
        self._held = ""

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        if self._withholding:
            self._held += delta
            return ""

        text = self._buffer + delta
        self._buffer = ""
        lower_text = text.lower()

        # 【核心优化】：如果文本中既没有 '[' 也没有 '`'，说明绝对不可能是工具调用，直接全部秒级放行！
        # 这样可以彻底根治普通正文、表格、结尾 Token 统计被缓冲区拦截或尾部截断的问题。
        if "[" not in lower_text and "`" not in lower_text:
            return text

        # 检查是否命中了真正的工具调用特征
        is_tool_block = False
        if "```json" in lower_text or "[called tool" in lower_text or "[caled tool" in lower_text:
            is_tool_block = True

        if is_tool_block:
            earliest_idx = -1
            for trigger in ("```json", "[called tool", "[CALLED TOOL"):
                idx = lower_text.find(trigger)
                if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
                    earliest_idx = idx
            
            if earliest_idx != -1:
                safe_prefix = text[:earliest_idx]
                self._withholding = True
                self._held = text[earliest_idx:]
                return safe_prefix

        # 如果包含 '[' 或 '`' 但还未构成完整工具特征，保留少量尾部防跨 chunk 切断
        tail_keep = 15
        if len(text) > tail_keep:
            safe_out = text[:-tail_keep]
            self._buffer = text[-tail_keep:]
            return safe_out

        self._buffer = text
        return ""

    def finish(self, discard_tool_call: bool) -> str:
        if self._withholding:
            return "" if discard_tool_call else self._held
        rem = self._buffer
        self._buffer = ""
        return rem


def build_tool_manifest(tools: list[dict] | None) -> str | None:
    """构建工具描述清单，加入严厉格式约束减少模型混合输出废话。"""
    if not tools:
        return None

    lines = [
        "System: You have access to the following tools. STRICT RULES:\n"
        "1. If calling one or more tools is the right next step, reply with NOTHING "
        "except a single fenced JSON block in this exact shape (no explanations before or after):\n"
        "```json\n"
        '{"tool_calls": [{"name": "<tool_name>", "arguments": { ... }}]}\n'
        "```\n"
        "2. Do NOT use [CALLED TOOL: ...] syntax and do NOT add conversational filler when calling tools.\n"
        "3. If no tool call is needed, answer normally in plain text without JSON blocks.",
        "Available tools:",
    ]
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        lines.append(f"- {name}: {desc}\n  parameters schema: {json.dumps(params, ensure_ascii=False)}")

    return "\n".join(lines)


def extract_tool_calls(reply_text: str) -> list[dict[str, Any]] | None:
    """双模工具提取器：同时解析标准 JSON 对象与 [CALLED TOOL / Called tool(s): name({...})] 混合文本。"""
    if not reply_text:
        return None

    # 1. 优先提取所有括号配对完整的 JSON 对象，寻找包含 "tool_calls" 的载荷
    for block in _extract_balanced_blocks(reply_text, "{", "}"):
        if "tool_calls" not in block:
            continue
        parsed = _parse_args_safely(block)
        raw_calls = parsed.get("tool_calls")
        if isinstance(raw_calls, list) and raw_calls:
            formatted = _format_openai_tool_calls(raw_calls)
            if formatted:
                return formatted

    # 2. 兼容解析 [CALLED TOOL: name({...})] 或 [Called tool(s): name({...})] 格式
    raw_called: list[dict[str, Any]] = []
    for match in _CALLED_HEADER_RE.finditer(reply_text):
        tool_name = match.group(1)
        # 从左圆括号 '(' 开始向后寻找配对的 ')'
        open_paren_pos = match.end() - 1
        paren_blocks = _extract_balanced_blocks(reply_text[open_paren_pos:], "(", ")")
        args_dict: dict[str, Any] = {}
        if paren_blocks:
            inner_args = paren_blocks[0][1:-1].strip()
            args_dict = _parse_args_safely(inner_args)
        raw_called.append({"name": tool_name, "arguments": args_dict})

    if raw_called:
        return _format_openai_tool_calls(raw_called)

    return None


def strip_tool_calls_from_text(reply_text: str) -> str:
    """从混合回复中剥离工具调用载荷，保留纯文本解释内容。"""
    cleaned = re.sub(r"```(?:json)?\s*\{.*?\}\s*```", "", reply_text, flags=re.DOTALL)
    cleaned = re.sub(r"\[(?:CALLED TOOL|Called tool\(s\)):.*?\]", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


# ==================== 本地沙箱执行工具集 ====================

def sync_write_file(path: str, content: str) -> str:
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote file to {path} ({len(content)} characters)."
    except Exception as e:
        return f"Error writing file to {path}: {str(e)}"


def sync_terminal(
    command: str,
    cwd: str,
    timeout: int = 600,
    background: bool = False,
    notify_on_complete: bool = False,
) -> str:
    try:
        if background:
            log_file = os.path.join(cwd, "bg_task.log")
            bg_cmd = f"nohup {command} > {log_file} 2>&1 &"
            subprocess.Popen(bg_cmd, shell=True, executable="/bin/bash", cwd=cwd)
            return f"Command started in background inside {cwd}: {command} (logs: {log_file})"
        result = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return f"Exit Code: {result.returncode}\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing terminal command: {str(e)}"


SERVER_TOOL_EXECUTORS = {
    "write_file": sync_write_file,
    "terminal": sync_terminal,
}