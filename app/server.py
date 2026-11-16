from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import shutil
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
from .tools import (
    SERVER_TOOL_EXECUTORS,
    ToolCallStreamFilter,
    build_tool_manifest,
    extract_tool_calls,
    strip_tool_calls_from_text,
)
from .upload import UploadError, upload_attachment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yun139-fastapi")

app = FastAPI(title="yun139-fastapi")
cfg = load_config()
pool = ClientPool(cfg.yun139.clients)

# 独立线程池执行阻塞 IO/Shell，避免冻结 FastAPI 主事件循环
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
SESSIONS_ROOT_DIR = "/opt/data/minimax/sessions"


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    stream: bool = True
    tools: list[dict] | None = None
    tool_choice: object | None = None


def _check_auth(authorization: str | None):
    if not cfg.server.api_key:
        return
    expected = f"Bearer {cfg.server.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _cleanup_old_sessions(max_age_seconds: int = 7200, interval_seconds: int = 3600):
    """后台定期清理超过存活时间的临时沙箱目录，防止磁盘占满。"""
    os.makedirs(SESSIONS_ROOT_DIR, exist_ok=True)
    while True:
        try:
            now = time.time()
            for entry in os.listdir(SESSIONS_ROOT_DIR):
                entry_path = os.path.join(SESSIONS_ROOT_DIR, entry)
                if os.path.isdir(entry_path):
                    try:
                        if now - os.path.getmtime(entry_path) > max_age_seconds:
                            shutil.rmtree(entry_path)
                            logger.info("cleaned expired sandbox directory: %s", entry_path)
                    except OSError:
                        continue
        except Exception as e:
            logger.error("sandbox cleanup loop error: %s", e)
        await asyncio.sleep(interval_seconds)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_cleanup_old_sessions())


@app.get("/v1/models")
async def list_models():
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


async def _execute_server_tool(
    func_name: str, raw_args: str, sandbox_dir: str, loop: asyncio.AbstractEventLoop
) -> str:
    """在请求级隔离沙箱内异步调度本地工具。"""
    if func_name not in SERVER_TOOL_EXECUTORS:
        return f"Error: Tool '{func_name}' is not registered on the server."

    try:
        args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        if not isinstance(args_dict, dict):
            args_dict = {}

        if func_name == "write_file":
            target_path = args_dict.get("path", "output.txt")
            if not os.path.isabs(target_path):
                target_path = os.path.join(sandbox_dir, target_path)
            content = args_dict.get("content", "")
            return await loop.run_in_executor(_executor, SERVER_TOOL_EXECUTORS["write_file"], target_path, content)

        if func_name == "terminal":
            command = args_dict.get("command", "")
            timeout = int(args_dict.get("timeout", 600))
            background = bool(args_dict.get("background", False))
            notify = bool(args_dict.get("notify_on_complete", False))
            return await loop.run_in_executor(
                _executor, SERVER_TOOL_EXECUTORS["terminal"], command, sandbox_dir, timeout, background, notify
            )

        return f"Error: Unhandled tool '{func_name}'."
    except Exception as e:
        logger.exception("tool execution failed: %s", func_name)
        return f"Error executing tool {func_name}: {str(e)}"


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, authorization: str | None = Header(None)):
    _check_auth(authorization)

    client_cfg = pool.next()
    id_ = sse.new_id()
    created = sse.now_ts()
    t0 = time.monotonic()

    # 为每个请求建立隔离沙箱工作目录，消除多并发文件覆盖冲突
    sandbox_dir = os.path.join(SESSIONS_ROOT_DIR, id_)
    os.makedirs(sandbox_dir, exist_ok=True)

    last_user_text = extract_last_user_text(body.messages)
    image_keywords = cfg.yun139.image_intent_keywords or DEFAULT_IMAGE_INTENT_KEYWORDS
    force_raw_image_request = (
        cfg.yun139.enable_image_generation
        and cfg.yun139.image_intent_bypass
        and last_user_text
        and looks_like_image_request(last_user_text, image_keywords)
    )

    tool_manifest = build_tool_manifest(body.tools) if cfg.tools.enabled and not force_raw_image_request else None
    current_messages = list(body.messages)

    attachment_field = None
    attachment_file_name = None
    if cfg.yun139.enable_attachments:
        extracted = extract_attachment_from_messages(current_messages)
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
                logger.info(
                    "waiting %.1fs before the chat request for upload propagation",
                    cfg.yun139.post_upload_delay_s,
                )
                await asyncio.sleep(cfg.yun139.post_upload_delay_s)
            attachment_field = build_attachment_field([uploaded])
            attachment_file_name = uploaded.name
            current_messages = [
                {"role": "user", "content": extract_last_user_text(current_messages) or "请描述并分析这个附件的内容。"}
            ]
            logger.info(
                "attachment uploaded (fileId=%s) — sending minimal dialogue instead of wrapped one",
                uploaded.file_id,
            )

    model_id = body.model or cfg.model.id
    loop = asyncio.get_running_loop()

    # 若未启用工具调用（或触发了纯生图直通），走原生单轮高速流式通道
    if not tool_manifest:
        dialogue = last_user_text if force_raw_image_request else build_dialogue(
            current_messages,
            None,
            cfg.yun139.max_chars_per_request,
            cfg.yun139.oversized_context_strategy,
        )
        logger.info("[%s] direct request start — dialogue_len=%d stream=%s", id_, len(dialogue), body.stream)

        if not body.stream:
            full_text = ""
            try:
                async for delta, is_final in stream_reply(
                    dialogue, client_cfg, cfg.yun139.upstream_url,
                    cfg.yun139.idle_timeout_s, cfg.yun139.hard_timeout_s,
                    cfg.yun139.enable_network_search, cfg.yun139.enable_image_generation,
                    attachment_field, attachment_file_name,
                ):
                    full_text += delta
                    if is_final:
                        break
            except UpstreamError as e:
                logger.exception("[%s] upstream error after %.1fs", id_, time.monotonic() - t0)
                return JSONResponse(status_code=502, content={"error": {"message": str(e)}})
            return sse.completion(id_, created, model_id, full_text, None)

        async def direct_event_stream():
            yield sse.sse(sse.chunk(id_, created, model_id, {"role": "assistant", "content": ""}, None))
            agen = stream_reply(
                dialogue, client_cfg, cfg.yun139.upstream_url,
                cfg.yun139.idle_timeout_s, cfg.yun139.hard_timeout_s,
                cfg.yun139.enable_network_search, cfg.yun139.enable_image_generation,
                attachment_field, attachment_file_name,
            )
            keepalive_s = cfg.yun139.keepalive_interval_s
            pending: asyncio.Task | None = None
            try:
                while True:
                    if pending is None:
                        pending = asyncio.ensure_future(agen.__anext__())
                    done, _ = await asyncio.wait({pending}, timeout=keepalive_s)
                    if pending not in done:
                        yield b": keepalive\n\n"
                        continue
                    task, pending = pending, None
                    try:
                        delta, is_final = task.result()
                    except StopAsyncIteration:
                        break
                    if delta:
                        yield sse.sse(sse.chunk(id_, created, model_id, {"content": delta}, None))
                    if is_final:
                        break
            except UpstreamError as e:
                yield sse.sse({"error": {"message": str(e)}})
                yield sse.sse_done()
                return
            yield sse.sse(sse.chunk(id_, created, model_id, {}, "stop"))
            yield sse.sse_done()

        return StreamingResponse(direct_event_stream(), media_type="text/event-stream")

    # ==================== 服务端自主 Agent 闭环（支持拦截混合文本并本地执行工具） ====================
    async def agent_loop_stream():
        if body.stream:
            yield sse.sse(sse.chunk(id_, created, model_id, {"role": "assistant", "content": ""}, None))

        max_turns = 10
        turn = 0
        final_reply_text = ""

        try:
            while turn < max_turns:
                turn += 1
                dialogue = build_dialogue(
                    current_messages,
                    tool_manifest,
                    cfg.yun139.max_chars_per_request,
                    cfg.yun139.oversized_context_strategy,
                )
                logger.info("[%s] Agent Turn %d start — dialogue_len=%d", id_, turn, len(dialogue))

                turn_full_text = ""
                stream_filter = ToolCallStreamFilter()
                agen = stream_reply(
                    dialogue, client_cfg, cfg.yun139.upstream_url,
                    cfg.yun139.idle_timeout_s, cfg.yun139.hard_timeout_s,
                    cfg.yun139.enable_network_search, cfg.yun139.enable_image_generation,
                    attachment_field if turn == 1 else None,
                    attachment_file_name if turn == 1 else None,
                )

                keepalive_s = cfg.yun139.keepalive_interval_s
                pending: asyncio.Task | None = None
                while True:
                    if pending is None:
                        pending = asyncio.ensure_future(agen.__anext__())
                    done, _ = await asyncio.wait({pending}, timeout=keepalive_s)
                    if pending not in done:
                        if body.stream:
                            yield b": keepalive\n\n"
                        continue

                    task, pending = pending, None
                    try:
                        delta, is_final = task.result()
                    except StopAsyncIteration:
                        break

                    if delta:
                        turn_full_text += delta
                        # 【终极直通】：普通对话流直接 100% 实时吐给客户端，绝对不经过任何过滤器的滞后缓存！
                        if body.stream:
                            yield sse.sse(sse.chunk(id_, created, model_id, {"content": delta}, None))
                    if is_final:
                        break

                # 统一在后台全文本中提取工具调用（双保险：不管前端怎么流，后端以内存全文本为准）
                tool_calls = extract_tool_calls(turn_full_text)
                
                # 【兜底直通车】：如果 extract_tool_calls 漏掉了，直接用正则强行把 terminal 和 write_file 抓出来！
                if not tool_calls and ("[Called tool" in turn_full_text or "[CALLED TOOL" in turn_full_text):
                    import re
                    matches = re.findall(r"\[(?:CALLED TOOL|Called tool\(s\)):\s*([a-zA-Z0-9_]+)\((.*?)\)\]", turn_full_text, re.DOTALL)
                    if matches:
                        tool_calls = []
                        for f_name, f_args_str in matches:
                            import json, ast
                            try:
                                args = json.loads(f_args_str)
                            except:
                                try:
                                    args = ast.literal_eval(f_args_str)
                                except:
                                    args = {"command": f_args_str}
                            tool_calls.append({
                                "id": "call_forced_" + uuid.uuid4().hex[:12],
                                "type": "function",
                                "function": {
                                    "name": f_name,
                                    "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                                }
                            })

                if not tool_calls:
                    # 【核心修复】：冲刷过滤器缓冲区残余文本，彻底解决尾部文字（如“是否立即”）被截断丢弃的问题
                    if stream_filter:
                        remaining_tail = stream_filter.finish(discard_tool_call=False)
                        if remaining_tail and body.stream:
                            yield sse.sse(sse.chunk(id_, created, model_id, {"content": remaining_tail}, None))
                    
                    final_reply_text = turn_full_text
                    break

                # 提取出属于服务器本地的工具
                server_handled_calls = [
                    c for c in tool_calls if c["function"]["name"] in SERVER_TOOL_EXECUTORS
                ]

                if not server_handled_calls:
                    # 如果模型调用的不是本地工具，正常透传给前端
                    if body.stream:
                        yield sse.sse(sse.chunk(id_, created, model_id, {"tool_calls": tool_calls}, None))
                        yield sse.sse(sse.chunk(id_, created, model_id, {}, "tool_calls"))
                        yield sse.sse_done()
                    return

                # ==================== 核心：在服务端真正执行本地工具并推进循环 ====================
                logger.info("[%s] Turn %d 成功拦截到本地工具调用: %s", id_, turn, [c["function"]["name"] for c in server_handled_calls])
                current_messages.append({"role": "assistant", "content": turn_full_text})

                for call in server_handled_calls:
                    func_name = call["function"]["name"]
                    raw_args = call["function"]["arguments"]
                    call_id = call["id"]

                    if body.stream:
                        status_hint = f"\n> ⚙️ [Backend Agent] 正在执行系统指令 `{func_name}`...\n"
                        yield sse.sse(sse.chunk(id_, created, model_id, {"content": status_hint}, None))

                    # 真正调用本地异步沙箱执行器！
                    tool_result = await _execute_server_tool(func_name, raw_args, sandbox_dir, loop)
                    logger.info("[%s] 本地工具 '%s' 执行完毕，返回结果长度: %d 字节", id_, func_name, len(tool_result))

                    # 将执行结果作为 tool 消息塞回对话历史，准备开启下一轮大模型推理
                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": func_name,
                            "content": tool_result,
                        }
                    )

            logger.info("[%s] Agent loop completed in %.1fs", id_, time.monotonic() - t0)
            if body.stream:
                yield sse.sse(sse.chunk(id_, created, model_id, {}, "stop"))
                yield sse.sse_done()
            else:
                clean_final = strip_tool_calls_from_text(final_reply_text)
                non_stream_holder.append(clean_final)

        except UpstreamError as e:
            logger.exception("[%s] upstream error after %.1fs", id_, time.monotonic() - t0)
            if body.stream:
                yield sse.sse({"error": {"message": str(e)}})
                yield sse.sse_done()
            else:
                non_stream_error.append(str(e))

    if body.stream:
        return StreamingResponse(agent_loop_stream(), media_type="text/event-stream")

    # 非流式调用模式复用同一套闭环逻辑
    non_stream_holder: list[str] = []
    non_stream_error: list[str] = []
    async for _ in agent_loop_stream():
        pass
    if non_stream_error:
        return JSONResponse(status_code=502, content={"error": {"message": non_stream_error[0]}})
    final_content = non_stream_holder[0] if non_stream_holder else ""
    return sse.completion(id_, created, model_id, final_content, None)