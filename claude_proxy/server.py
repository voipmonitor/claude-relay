"""aiohttp server: POST /v1/messages handler."""

import json
import os
import logging
import re

import aiohttp
from aiohttp import web

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug")

from .config import ProxyConfig
from .convert_request import convert_request
from .convert_stream import StreamResult, convert_openai_stream_to_anthropic
from .sse import parse_sse_stream
from .backend import send_to_backend, detect_backend, get_state
from .image_agent import has_images, strip_and_cache_images, image_agent_stream
from .normalize import normalize_request

log = logging.getLogger(__name__)


def _summarize_messages(messages: list) -> str:
    """One-line summary of message roles and content types."""
    parts = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(f"{role}(text:{len(content)})")
        elif isinstance(content, list):
            types = []
            for b in content:
                bt = b.get("type", "?")
                if bt == "image":
                    types.append("img")
                elif bt == "tool_result":
                    sub = b.get("content")
                    has_img = False
                    if isinstance(sub, list):
                        has_img = any(isinstance(x, dict) and x.get("type") == "image" for x in sub)
                    types.append(f"tool_result{'(img)' if has_img else ''}")
                elif bt == "tool_use":
                    types.append(f"tool_use:{b.get('name', '?')}")
                elif bt == "thinking":
                    types.append("think")
                elif bt == "text":
                    types.append(f"text:{len(b.get('text', ''))}")
                else:
                    types.append(bt)
            parts.append(f"{role}[{','.join(types)}]")
        else:
            parts.append(role)
    return " | ".join(parts[-6:])  # last 6 messages


async def handle_messages(request: web.Request) -> web.StreamResponse:
    config: ProxyConfig = request.app["config"]
    session: aiohttp.ClientSession = request.app["session"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    # Session ID for image agent (from metadata.user_id as Claude Code sends)
    session_id = (body.get("metadata") or {}).get("user_id", "default")
    use_image_agent = config.image_agent_enabled and has_images(body)
    req_id = os.urandom(3).hex()

    msgs = body.get("messages", [])
    num_tools = len(body.get("tools", []))
    tool_names = [t.get("name", "?") for t in body.get("tools", [])]
    system_len = len(body.get("system", "")) if isinstance(body.get("system"), str) else sum(
        len(b.get("text", "")) for b in body.get("system", []) if isinstance(b, dict)
    )
    thinking = body.get("thinking", {})
    thinking_info = f"budget={thinking.get('budget_tokens')}" if thinking.get("type") == "enabled" else "off"

    log.info(
        "[%s] >>> POST /v1/messages msgs=%d tools=%d(%s) system=%d thinking=%s image_agent=%s",
        req_id, len(msgs), num_tools,
        ",".join(tool_names[:5]) + ("..." if len(tool_names) > 5 else ""),
        system_len, thinking_info, use_image_agent,
    )
    log.info("[%s]     last_msgs: %s", req_id, _summarize_messages(msgs))

    # Dump full request body for debugging
    try:
        debug_file = os.path.join(DEBUG_DIR, f"{req_id}_anthropic.json")
        with open(debug_file, "w") as f:
            json.dump(body, f, indent=2, ensure_ascii=False, default=str)
        log.info("[%s]     dumped anthropic body to %s", req_id, debug_file)
    except Exception as e:
        log.warning("[%s]     failed to dump body: %s", req_id, e)

    if use_image_agent:
        strip_and_cache_images(body, session_id)

    # Normalize request for KV cache stability
    body, norm_changes = normalize_request(body, config)
    if norm_changes:
        log.info("[%s]     normalize: %s", req_id, "; ".join(norm_changes))

    # Convert Anthropic → OpenAI
    openai_body = convert_request(body)

    # Force analyzeImage tool_choice when image agent is active
    if use_image_agent and config.force_vision:
        openai_body["tool_choice"] = {"type": "function", "function": {"name": "analyzeImage"}}
        log.info("[%s]     force_vision: tool_choice set to analyzeImage", req_id)

    openai_msgs = openai_body.get("messages", [])
    log.info("[%s]     openai: %d msgs, model=%s, max_tokens=%s, tools=%d",
             req_id, len(openai_msgs), openai_body.get("model"),
             openai_body.get("max_completion_tokens"),
             len(openai_body.get("tools", [])))

    # Dump converted OpenAI body for debugging
    try:
        debug_file = os.path.join(DEBUG_DIR, f"{req_id}_openai.json")
        with open(debug_file, "w") as f:
            json.dump(openai_body, f, indent=2, ensure_ascii=False, default=str)
        log.info("[%s]     dumped openai body to %s", req_id, debug_file)
    except Exception as e:
        log.warning("[%s]     failed to dump openai body: %s", req_id, e)

    # Count input tokens via tiktoken for message_start
    input_token_count = len(_get_tiktoken().encode(_serialize_for_counting(body)))

    # Send main streaming request to backend (with auto-retry on context overflow)
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            resp = await send_to_backend(session, config, openai_body, req_id=req_id)
        except Exception as e:
            log.error("[%s] Backend request failed: %s", req_id, e)
            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": str(e)}},
                status=502,
            )

        if resp.status != 200:
            error_body = await resp.text()
            log.error("[%s] Backend returned %d: %s", req_id, resp.status, error_body[:500])
            count_task.cancel()
            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": error_body}},
                status=resp.status,
            )

        log.info("[%s]     backend responded 200, streaming...", req_id)

        # For non-image-agent: peek at first SSE events to detect errors/empty responses
        if not use_image_agent:
            sse_events = parse_sse_stream(resp)
            buffered = []
            has_content = False
            async for event in sse_events:
                buffered.append(event)
                if event.data == "[DONE]":
                    break
                try:
                    data = json.loads(event.data)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta:
                            has_content = True
                            break
                except (json.JSONDecodeError, TypeError):
                    pass

            if not has_content:
                # Check if backend returned a context-overflow error
                error_msg = "Backend returned empty response"
                context_overflow = False
                for ev in buffered:
                    if ev.data == "[DONE]":
                        continue
                    try:
                        ev_data = json.loads(ev.data)
                        if "error" in ev_data:
                            error_msg = ev_data["error"].get("message", str(ev_data["error"]))
                            m = re.search(r"(\d+) tokens from the input.*?(\d+) tokens for the completion", error_msg)
                            if m:
                                context_overflow = True
                                input_tokens = int(m.group(1))
                                m2 = re.search(r"maximum context length of (\d+)", error_msg)
                                ctx_limit = int(m2.group(1)) if m2 else config.context_limit
                                break
                    except (json.JSONDecodeError, TypeError):
                        pass

                if context_overflow and attempt < max_retries:
                    new_max = max(config.min_completion_tokens, ctx_limit - input_tokens - 100)
                    log.warning("[%s] Context overflow: input=%d, limit=%d → retrying with max_completion_tokens=%d (attempt %d)",
                                req_id, input_tokens, ctx_limit, new_max, attempt + 1)
                    openai_body["max_completion_tokens"] = new_max
                    continue

                log.error("[%s] EMPTY/ERROR response from backend — got %d events, error: %s",
                          req_id, len(buffered), error_msg)
                for i, ev in enumerate(buffered):
                    log.error("[%s]   event[%d]: %s", req_id, i, ev.data[:500])
                return web.json_response(
                    {"type": "error", "error": {"type": "overloaded_error",
                     "message": error_msg}},
                    status=529,
                )

            async def _replay_and_continue():
                for ev in buffered:
                    yield ev
                async for ev in sse_events:
                    yield ev

            sse_source = _replay_and_continue()
        else:
            sse_source = None

        break  # success

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)

    bytes_sent = 0
    try:
        if use_image_agent:
            async for chunk in image_agent_stream(resp, openai_body, session_id, session, config, req_id):
                await response.write(chunk)
                bytes_sent += len(chunk)
        else:
            async for chunk in convert_openai_stream_to_anthropic(sse_source, req_id=req_id, input_tokens=input_token_count):
                await response.write(chunk)
                bytes_sent += len(chunk)
    except Exception as e:
        log.error("[%s] Streaming error: %s", req_id, e, exc_info=True)

    log.info("[%s] <<< done, %d bytes sent", req_id, bytes_sent)
    await response.write_eof()
    return response


def _serialize_for_counting(body: dict) -> str:
    """Serialize Anthropic count_tokens body to plain text for tokenization."""
    parts = []

    # System prompt
    system = body.get("system")
    if isinstance(system, str):
        parts.append(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))

    # Messages
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                bt = block.get("type", "")
                if bt == "text":
                    parts.append(block.get("text", ""))
                elif bt == "tool_use":
                    parts.append(json.dumps(block.get("input", {}), ensure_ascii=False))
                elif bt == "tool_result":
                    sub = block.get("content", "")
                    if isinstance(sub, str):
                        parts.append(sub)
                    elif isinstance(sub, list):
                        for sb in sub:
                            if isinstance(sb, dict):
                                if sb.get("type") == "text":
                                    parts.append(sb.get("text", ""))
                                elif sb.get("type") == "image":
                                    parts.append("[image]")  # ~1600 tokens per image, rough estimate
                elif bt == "thinking":
                    parts.append(block.get("thinking", ""))

    # Tools
    for tool in body.get("tools", []):
        parts.append(tool.get("name", ""))
        parts.append(tool.get("description", ""))
        schema = tool.get("input_schema")
        if schema:
            parts.append(json.dumps(schema, ensure_ascii=False))

    return "\n".join(parts)


_tiktoken_enc = None

def _get_tiktoken():
    global _tiktoken_enc
    if _tiktoken_enc is None:
        import tiktoken
        _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_enc


async def handle_count_tokens(request: web.Request) -> web.Response:
    """POST /v1/messages/count_tokens — token counting for Claude Code /context."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    text = _serialize_for_counting(body)
    enc = _get_tiktoken()
    count = len(enc.encode(text))
    # Debug: log what we're counting
    num_msgs = len(body.get("messages", []))
    num_tools = len(body.get("tools", []))
    sys_len = len(body.get("system", "")) if isinstance(body.get("system"), str) else sum(
        len(b.get("text", "")) for b in body.get("system", []) if isinstance(b, dict)
    )
    log.debug("count_tokens: msgs=%d tools=%d system=%d text_len=%d → %d tokens",
              num_msgs, num_tools, sys_len, len(text), count)
    return web.json_response({"input_tokens": count})


async def health_check(request: web.Request) -> web.Response:
    state = get_state()
    config: ProxyConfig = request.app["config"]
    return web.json_response({
        **state.info(),
        "backend_url": config.backend_url,
        "image_agent_enabled": config.image_agent_enabled,
        "vision_url": config.vision_url,
        "vision_model": config.vision_model,
    })


async def on_startup(app: web.Application):
    app["session"] = aiohttp.ClientSession()
    config: ProxyConfig = app["config"]
    await detect_backend(app["session"], config.backend_url)


async def on_cleanup(app: web.Application):
    await app["session"].close()


def create_app(config: ProxyConfig) -> web.Application:
    app = web.Application(client_max_size=config.client_max_size)
    app["config"] = config
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/v1/messages", handle_messages)
    app.router.add_post("/v1/messages/count_tokens", handle_count_tokens)
    app.router.add_get("/health", health_check)
    return app
