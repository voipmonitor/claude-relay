"""aiohttp server: POST /v1/messages handler."""

import asyncio
import errno
import json
import os
import logging
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import AsyncIterator

import aiohttp
from aiohttp import web

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug")

from .config import ProxyConfig
from .convert_request import convert_request
from .convert_stream import StreamResult, convert_openai_stream_to_anthropic
from .sse import SSEEvent, make_anthropic_sse, parse_sse_stream
from .backend import send_to_backend, detect_backend, get_state, get_states_info
from .image_agent import has_images, strip_and_cache_images, image_agent_stream, ImageCache
from .normalize import normalize_request

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextOverflowRetry:
    max_completion_tokens: int
    ctx_limit: int
    input_tokens: int | None
    reason: str
    input_tokens_is_lower_bound: bool = False


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


def validate_request(body: dict, config: ProxyConfig) -> None:
    """Validate Anthropic API request body."""
    # Messages validation
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise web.HTTPBadRequest(text='{"error":"messages must be a non-empty array"}')

    # Max tokens validation
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise web.HTTPBadRequest(text='{"error":"max_tokens must be a positive integer"}')

    # Tools length validation
    tools = body.get("tools", [])
    if len(tools) > config.max_tools:
        raise web.HTTPBadRequest(text=f'{{"error":"tools array exceeds maximum length of {config.max_tools}"}}')

    # Per-image size validation
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image":
                    src = block.get("source", {})
                    data = src.get("data", "")
                    if len(data) > config.max_image_b64_chars:
                        raise web.HTTPRequestEntityTooLarge(
                            text=f'{{"error":"image exceeds maximum size of {config.max_image_b64_chars} characters"}}'
                        )


def _tool_names(tools: list[dict]) -> list[str]:
    names = []
    for tool in tools:
        if isinstance(tool, dict):
            names.append(str(tool.get("name") or "?"))
    return names


def _openai_tool_names(tools: list[dict]) -> list[str]:
    names = []
    for tool in tools:
        if isinstance(tool, dict):
            func = tool.get("function") or {}
            names.append(str(func.get("name") or "?"))
    return names


def _parse_token_count(value: str) -> int:
    return int(value.replace(",", ""))


class ContextWindowError(ValueError):
    """Raised when input tokens alone exceed the context window."""


_CONTEXT_RETRY_MARGIN = 100
_CONTEXT_LOWER_BOUND_RETRY_MARGIN = 1024


def _context_retry_completion_budget(
    ctx_limit: int,
    input_tokens: int | None,
    config: ProxyConfig,
    *,
    input_tokens_is_lower_bound: bool = False,
) -> int:
    # vLLM's "prompt contains at least ..." count is a lower bound observed at
    # overflow, not a stable exact prompt token count. Use a larger reserve for
    # that form so retries do not chase a moving one-token-over boundary.
    margin = _CONTEXT_LOWER_BOUND_RETRY_MARGIN if input_tokens_is_lower_bound else _CONTEXT_RETRY_MARGIN
    available_tokens = ctx_limit - margin
    if input_tokens is not None:
        available_tokens -= input_tokens
    return max(config.min_completion_tokens, available_tokens)


def _completion_token_margin(local_input_tokens: int) -> int:
    """Small fixed reserve. The 400-retry path corrects precisely when tiktoken underestimates."""
    return 128


def _cap_max_completion_tokens(
    requested_max: int,
    *,
    ctx_limit: int,
    input_tokens: int,
) -> tuple[int, int]:
    margin = _completion_token_margin(input_tokens)
    available = ctx_limit - input_tokens - margin
    if available <= 0:
        raise ContextWindowError(
            f"Input exceeds context window: input_tokens={input_tokens}, "
            f"ctx_limit={ctx_limit}, margin={margin}"
        )
    capped = min(requested_max, available)
    return capped, margin


def _context_overflow_retry_from_error(
    error_body: str,
    config: ProxyConfig,
    estimated_input_tokens: int | None = None,
) -> ContextOverflowRetry | None:
    """Detect non-streaming backend context errors and compute the retry budget."""
    flags = re.IGNORECASE | re.DOTALL

    # vLLM/SGLang output-only limit:
    # "max_completion_tokens=X cannot be greater than max_model_len=Y"
    match = re.search(r"max_completion_tokens\s*=\s*([\d,]+).*?max_model_len\D*([\d,]+)", error_body, flags)
    if match:
        ctx_limit = _parse_token_count(match.group(2))
        return ContextOverflowRetry(
            max_completion_tokens=_context_retry_completion_budget(ctx_limit, estimated_input_tokens, config),
            ctx_limit=ctx_limit,
            input_tokens=estimated_input_tokens,
            reason="completion_limit",
        )

    # vLLM combined input+output limit:
    # "maximum context length is X. However, you requested Y output tokens and your prompt contains Z input tokens"
    match = re.search(
        r"maximum context length is\s*([\d,]+).*?"
        r"requested\s*([\d,]+)\s*output tokens.*?"
        r"prompt contains(?P<lower_bound>\s+at least)?\s*([\d,]+)\s*input tokens",
        error_body,
        flags,
    )
    if match:
        ctx_limit = _parse_token_count(match.group(1))
        input_tokens = _parse_token_count(match.group(4))
        input_tokens_is_lower_bound = bool(match.group("lower_bound"))
        return ContextOverflowRetry(
            max_completion_tokens=_context_retry_completion_budget(
                ctx_limit,
                input_tokens,
                config,
                input_tokens_is_lower_bound=input_tokens_is_lower_bound,
            ),
            ctx_limit=ctx_limit,
            input_tokens=input_tokens,
            reason="context_limit",
            input_tokens_is_lower_bound=input_tokens_is_lower_bound,
        )

    if not match:
        # OpenAI-compatible/SGLang shape:
        # "maximum context length is X tokens. However, you requested Y tokens (Z in the messages, W in the completion)."
        match = re.search(
            r"maximum context length is\s*([\d,]+).*?"
            r"requested\s*([\d,]+)\s*tokens.*?"
            r"\(([\d,]+)\s*in (?:the )?(?:messages|prompt).*?([\d,]+)\s*in (?:the )?(?:completion|output)",
            error_body,
            flags,
        )
    if not match:
        # vLLM/SGLang newer shape:
        # "Requested token count exceeds the model's maximum context length of X tokens.
        #  You requested a total of Y tokens: Z tokens from the input messages and W tokens for the completion."
        match = re.search(
            r"maximum context length of\s*([\d,]+)\s*tokens.*?"
            r"requested a total of\s*([\d,]+)\s*tokens.*?"
            r"([\d,]+)\s*tokens from (?:the )?(?:input messages|messages|prompt).*?"
            r"([\d,]+)\s*tokens for (?:the )?(?:completion|output)",
            error_body,
            flags,
        )

    if match:
        ctx_limit = _parse_token_count(match.group(1))
        input_tokens = _parse_token_count(match.group(3))
        return ContextOverflowRetry(
            max_completion_tokens=_context_retry_completion_budget(ctx_limit, input_tokens, config),
            ctx_limit=ctx_limit,
            input_tokens=input_tokens,
            reason="context_limit",
        )

    return None


_DISCONNECT_EXC = (ConnectionResetError, aiohttp.ClientConnectionResetError, BrokenPipeError)

KEEPALIVE_INTERVAL = 5  # seconds between SSE keepalive comments

_EMPTY_RESPONSE_RETRY_PROMPT = (
    "The previous assistant completion ended without visible assistant text or "
    "a tool call. Continue the current task now. If hidden thinking is open, "
    "close the thinking block before the final answer. Put the final "
    "user-facing answer in normal visible assistant content, not hidden "
    "reasoning. If a tool is needed, call a tool instead."
)


class _ClientGone(Exception):
    """Raised when a keepalive write fails because the client disconnected."""


class _ResponseDumper:
    """Best-effort dump of bytes sent to the Anthropic client."""

    def __init__(self, req_id: str, enabled: bool):
        self.req_id = req_id
        self.path: str | None = None
        self._file = None
        if not enabled:
            return

        os.makedirs(DEBUG_DIR, exist_ok=True)
        self.path = os.path.join(DEBUG_DIR, f"{req_id}_anthropic_stream.sse")
        try:
            self._file = open(self.path, "wb")
            log.info("[%s] response_dump: dumping Anthropic SSE to %s", req_id, self.path)
        except OSError as exc:
            log.warning("[%s] response_dump: failed to open %s: %s", req_id, self.path, exc)
            self.path = None

    def write(self, chunk: bytes) -> None:
        if self._file is None:
            return
        try:
            self._file.write(chunk)
            self._file.flush()
        except OSError as exc:
            log.warning("[%s] response_dump: failed to write %s: %s", self.req_id, self.path, exc)
            self.close()

    def close(self) -> None:
        if self._file is None:
            return
        try:
            self._file.close()
        except OSError:
            pass
        finally:
            self._file = None


def _is_disconnect_error(exc: BaseException) -> bool:
    if isinstance(exc, _DISCONNECT_EXC):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in {
        errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED,
    }:
        return True
    return "closing transport" in str(exc).lower()


async def _send_keepalive(
    response: web.StreamResponse,
    response_dump: _ResponseDumper | None = None,
) -> None:
    """Send an SSE comment keepalive. Raises _ClientGone if client disconnected."""
    chunk = b": keepalive\n\n"
    try:
        await response.write(chunk)
        if response_dump is not None:
            response_dump.write(chunk)
    except _DISCONNECT_EXC as exc:
        raise _ClientGone from exc


async def _emit_sse_error(
    response: web.StreamResponse,
    error_type: str,
    message: str,
    response_dump: _ResponseDumper | None = None,
) -> None:
    """Emit an SSE error event and close the stream. For post-prepare error paths."""
    payload = json.dumps({"type": "error", "error": {"type": error_type, "message": message}})
    chunk = f"event: error\ndata: {payload}\n\n".encode()
    try:
        await response.write(chunk)
        if response_dump is not None:
            response_dump.write(chunk)
    except _DISCONNECT_EXC:
        pass
    try:
        await response.write_eof()
    except (RuntimeError, *_DISCONNECT_EXC):
        pass


async def _prepare_stream_response(request: web.Request) -> web.StreamResponse | None:
    """Create and prepare a StreamResponse. Returns None if client already disconnected."""
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    try:
        await response.prepare(request)
    except _DISCONNECT_EXC:
        log.warning("Client disconnected before stream prepare")
        return None
    return response


def _delta_has_visible_output(delta: dict) -> bool:
    """Return True when an OpenAI delta contains user-visible output."""
    content = delta.get("content")
    if isinstance(content, str):
        if content.strip():
            return True
    elif content:
        return True

    if delta.get("tool_calls"):
        return True

    return False


def _append_empty_response_retry_instruction(openai_body: dict) -> None:
    messages = openai_body.setdefault("messages", [])
    if messages and messages[-1].get("content") == _EMPTY_RESPONSE_RETRY_PROMPT:
        return
    messages.append({"role": "user", "content": _EMPTY_RESPONSE_RETRY_PROMPT})


async def _peek_with_keepalive(
    sse_events: AsyncIterator[SSEEvent],
    response: web.StreamResponse,
    response_dump: _ResponseDumper | None = None,
    interval: int = KEEPALIVE_INTERVAL,
) -> tuple[list[SSEEvent], bool]:
    """Buffer SSE events until first client-visible output or [DONE], sending keepalives during waits.

    Returns (buffered_events, has_visible_output).
    Raises _ClientGone if the client disconnects during peek.
    """
    buffered = []
    has_visible_output = False
    ait = sse_events.__aiter__()
    pending = asyncio.ensure_future(ait.__anext__())

    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                await _send_keepalive(response, response_dump)
                continue

            try:
                event = pending.result()
            except StopAsyncIteration:
                break
            except Exception as exc:
                # Backend-read exception — don't conflate with client disconnect.
                # Let the caller decide how to handle backend failures.
                log.exception("Error while peeking SSE stream")
                raise

            buffered.append(event)
            if event.data == "[DONE]":
                break

            try:
                data = json.loads(event.data)
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if _delta_has_visible_output(delta):
                        has_visible_output = True
                        break
            except (json.JSONDecodeError, TypeError):
                pass

            pending = asyncio.ensure_future(ait.__anext__())
    finally:
        if not pending.done():
            pending.cancel()
        try:
            await pending
        except (StopAsyncIteration, asyncio.CancelledError, Exception):
            pass

    return buffered, has_visible_output


def _tool_state_summary(tool_states: dict[str, dict]) -> list[dict]:
    summary = []
    for index, state in sorted(tool_states.items(), key=lambda item: item[0]):
        args = state.get("arguments", "")
        item = {
            "index": index,
            "id": state.get("id"),
            "name": state.get("name"),
            "chunks": state.get("chunks", 0),
            "arguments_chars": len(args),
            "arguments_preview": args[:500],
        }
        if args:
            try:
                json.loads(args)
                item["arguments_json_valid"] = True
            except json.JSONDecodeError as e:
                item["arguments_json_valid"] = False
                item["arguments_json_error"] = str(e)
        summary.append(item)
    return summary


async def _debug_sse_stream(
    sse_events: AsyncIterator[SSEEvent],
    req_id: str,
    *,
    enabled: bool,
    expected_tool_names: list[str] | None = None,
) -> AsyncIterator[SSEEvent]:
    """Optionally dump raw backend SSE and summarize tool-call parser signals."""
    if not enabled:
        async for event in sse_events:
            yield event
        return

    os.makedirs(DEBUG_DIR, exist_ok=True)
    stream_path = os.path.join(DEBUG_DIR, f"{req_id}_backend_stream.ndjson")
    summary_path = os.path.join(DEBUG_DIR, f"{req_id}_tool_debug.json")
    delta_keys: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    unknown_tool_names: Counter[str] = Counter()
    tail = deque(maxlen=12)
    counts = Counter()
    content_chars = 0
    reasoning_chars = 0
    tool_states: dict[str, dict] = {}
    tool_like_content = []
    expected_tools = set(expected_tool_names or [])
    stream_file = None

    try:
        stream_file = open(stream_path, "w")
        log.info("[%s] tool_debug: dumping backend SSE to %s", req_id, stream_path)
    except OSError as e:
        log.warning("[%s] tool_debug: failed to open stream dump %s: %s", req_id, stream_path, e)

    try:
        async for event in sse_events:
            counts["events"] += 1
            seq = counts["events"]
            record = {
                "seq": seq,
                "event": event.event,
                "raw": event.data,
            }

            if event.data == "[DONE]":
                counts["done"] += 1
                tail.append({"seq": seq, "done": True})
                if stream_file:
                    stream_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                yield event
                continue

            event_summary = {"seq": seq, "event": event.event}
            try:
                data = json.loads(event.data)
            except (json.JSONDecodeError, TypeError) as e:
                counts["bad_json"] += 1
                event_summary["bad_json"] = str(e)
                tail.append(event_summary)
                if stream_file:
                    stream_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                yield event
                continue

            choices = data.get("choices") or []
            if choices:
                choice = choices[0] or {}
                delta = choice.get("delta") or {}
                finish_reason = choice.get("finish_reason")
                keys = sorted(delta.keys())
                for key in keys:
                    delta_keys[key] += 1

                if finish_reason:
                    finish_reasons[str(finish_reason)] += 1

                content = delta.get("content")
                if content:
                    content_text = str(content)
                    content_chars += len(content_text)
                    event_summary["content_preview"] = content_text[:200]
                    lower_content = content_text.lower()
                    looks_like_tool = any(
                        marker in lower_content
                        for marker in ("tool_call", "function_call", "<tool", "<function", '"arguments"')
                    )
                    if not looks_like_tool and expected_tools:
                        looks_like_tool = any(f"{name}(" in content_text for name in expected_tools)
                    if looks_like_tool:
                        counts["tool_like_content_events"] += 1
                        tool_like_content.append({
                            "seq": seq,
                            "preview": content_text[:300],
                        })

                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if not reasoning and isinstance(delta.get("thinking"), dict):
                    reasoning = delta["thinking"].get("content")
                if reasoning:
                    reasoning_text = str(reasoning)
                    reasoning_chars += len(reasoning_text)
                    event_summary["reasoning_preview"] = reasoning_text[:200]

                tool_calls = delta.get("tool_calls") or []
                if tool_calls:
                    counts["tool_call_events"] += 1
                    counts["tool_call_deltas"] += len(tool_calls)
                    event_summary["tool_calls"] = []
                    for tool_call in tool_calls:
                        func = (tool_call or {}).get("function") or {}
                        name = func.get("name")
                        if name:
                            tool_names[str(name)] += 1
                            if expected_tools and str(name) not in expected_tools:
                                unknown_tool_names[str(name)] += 1
                        args = func.get("arguments")
                        index = str(tool_call.get("index", 0))
                        state = tool_states.setdefault(index, {
                            "id": None,
                            "name": None,
                            "arguments": "",
                            "chunks": 0,
                        })
                        if tool_call.get("id"):
                            state["id"] = tool_call.get("id")
                        if name:
                            state["name"] = name
                        state["chunks"] += 1
                        if isinstance(args, str):
                            state["arguments"] += args
                        event_summary["tool_calls"].append({
                            "index": tool_call.get("index"),
                            "id": tool_call.get("id"),
                            "name": name,
                            "args_chars": len(args) if isinstance(args, str) else 0,
                            "args_preview": args[:200] if isinstance(args, str) else None,
                        })

                event_summary["delta_keys"] = keys
                event_summary["finish_reason"] = finish_reason
                tail.append(event_summary)

                if finish_reason or tool_calls:
                    log.info(
                        "[%s] tool_debug: event #%d finish=%s keys=%s tool_call_deltas=%d",
                        req_id,
                        seq,
                        finish_reason,
                        ",".join(keys),
                        len(tool_calls),
                    )

            if stream_file:
                stream_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            yield event
    finally:
        if stream_file:
            stream_file.close()

        summary = {
            "request_id": req_id,
            "events": counts["events"],
            "done_events": counts["done"],
            "bad_json_events": counts["bad_json"],
            "tool_call_events": counts["tool_call_events"],
            "tool_call_deltas": counts["tool_call_deltas"],
            "tool_names": dict(tool_names),
            "expected_tool_names": sorted(expected_tools),
            "unknown_tool_names": dict(unknown_tool_names),
            "tool_calls_by_index": _tool_state_summary(tool_states),
            "finish_reasons": dict(finish_reasons),
            "delta_keys": dict(delta_keys),
            "content_chars": content_chars,
            "reasoning_chars": reasoning_chars,
            "tool_like_content_events": counts["tool_like_content_events"],
            "tool_like_content": tool_like_content[-5:],
            "stream_path": stream_path if stream_file else None,
            "tail": list(tail),
        }

        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except OSError as e:
            log.warning("[%s] tool_debug: failed to write summary %s: %s", req_id, summary_path, e)

        if finish_reasons.get("tool_calls", 0) and counts["tool_call_deltas"] == 0:
            log.error(
                "[%s] tool_debug: backend reported finish_reason=tool_calls but emitted no delta.tool_calls; summary=%s",
                req_id,
                summary_path,
            )
        elif unknown_tool_names:
            log.error(
                "[%s] tool_debug: backend emitted unknown tool name(s) %s; expected=%s summary=%s",
                req_id,
                dict(unknown_tool_names),
                sorted(expected_tools),
                summary_path,
            )
        elif any(item.get("arguments_json_valid") is False for item in summary["tool_calls_by_index"]):
            log.error("[%s] tool_debug: backend emitted invalid tool argument JSON; summary=%s", req_id, summary_path)
        elif counts["tool_like_content_events"] and counts["tool_call_deltas"] == 0:
            log.warning(
                "[%s] tool_debug: backend emitted tool-looking plain content but no delta.tool_calls; summary=%s",
                req_id,
                summary_path,
            )
        else:
            log.info(
                "[%s] tool_debug: events=%d tool_call_deltas=%d finish=%s summary=%s",
                req_id,
                counts["events"],
                counts["tool_call_deltas"],
                dict(finish_reasons),
                summary_path,
            )


async def handle_messages(request: web.Request) -> web.StreamResponse:
    config: ProxyConfig = request.app["config"]
    session: aiohttp.ClientSession = request.app["session"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    validate_request(body, config)

    # Session ID for image agent: use user_id if provided, otherwise req_id (not "default") to prevent collisions
    req_id = os.urandom(3).hex()
    session_id = (body.get("metadata") or {}).get("user_id", req_id)
    use_image_agent = config.image_agent_enabled and has_images(body)

    msgs = body.get("messages", [])
    num_tools = len(body.get("tools", []))
    tool_names = _tool_names(body.get("tools", []))
    system_len = len(body.get("system", "")) if isinstance(body.get("system"), str) else sum(
        len(b.get("text", "")) for b in body.get("system", []) if isinstance(b, dict)
    )
    thinking = body.get("thinking", {})
    thinking_enabled = isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive")
    thinking_info = f"budget={thinking.get('budget_tokens')}" if thinking_enabled else "off"

    log.info(
        "[%s] >>> POST /v1/messages msgs=%d tools=%d(%s) system=%d thinking=%s image_agent=%s",
        req_id, len(msgs), num_tools,
        ",".join(tool_names[:5]) + ("..." if len(tool_names) > 5 else ""),
        system_len, thinking_info, use_image_agent,
    )
    log.info("[%s]     last_msgs: %s", req_id, _summarize_messages(msgs))

    # Dump full request body for debugging (opt-in via --dump-requests)
    if config.dump_requests:
        try:
            debug_file = os.path.join(DEBUG_DIR, f"{req_id}_anthropic.json")
            with open(debug_file, "w") as f:
                json.dump(body, f, indent=2, ensure_ascii=False, default=str)
            log.info("[%s]     dumped anthropic body to %s", req_id, debug_file)
        except Exception as e:
            log.warning("[%s]     failed to dump body: %s", req_id, e)

    if use_image_agent:
        strip_and_cache_images(body, session_id, request.app["image_cache"])

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
    openai_tool_names = _openai_tool_names(openai_body.get("tools", []))
    tool_debug_enabled = config.tool_debug and bool(openai_tool_names)
    backend_target = config.resolve_backend(str(openai_body.get("model") or "auto"))
    state = await detect_backend(session, backend_target.backend_url)
    log.info(
        "[%s]     route: model=%s -> backend=%s route=%s upstream_model=%s",
        req_id,
        backend_target.request_model,
        backend_target.backend_url,
        backend_target.route_name,
        backend_target.upstream_model or state.model,
    )
    log.info("[%s]     openai: %d msgs, model=%s, max_tokens=%s, tools=%d tool_choice=%s",
             req_id, len(openai_msgs), openai_body.get("model"),
             openai_body.get("max_completion_tokens"),
             len(openai_tool_names), openai_body.get("tool_choice"))
    if openai_tool_names:
        log.info(
            "[%s]     openai_tools: %s%s",
            req_id,
            ",".join(openai_tool_names[:12]),
            "..." if len(openai_tool_names) > 12 else "",
        )
    if tool_debug_enabled:
        log.info("[%s]     tool_debug enabled for this request", req_id)

    if config.dump_requests:
        try:
            debug_file = os.path.join(DEBUG_DIR, f"{req_id}_openai.json")
            with open(debug_file, "w") as f:
                json.dump(openai_body, f, indent=2, ensure_ascii=False, default=str)
            log.info("[%s]     dumped openai body to %s", req_id, debug_file)
        except Exception as e:
            log.warning("[%s]     failed to dump openai body: %s", req_id, e)

    # Count input tokens via tiktoken for capping and message_start
    input_token_count = len(_get_tiktoken().encode(_serialize_for_counting(body)))
    requested_max = openai_body.get("max_completion_tokens")
    if isinstance(requested_max, int) and requested_max > 0:
        try:
            capped_max, margin = _cap_max_completion_tokens(
                requested_max,
                ctx_limit=state.context_limit,
                input_tokens=input_token_count,
            )
        except ContextWindowError as exc:
            log.error("[%s] Context window exceeded: %s", req_id, exc)
            return web.json_response(
                {"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}},
                status=400,
            )
        if capped_max < requested_max:
            log.warning(
                "[%s] Capping max_completion_tokens %d → %d (input≈%d, limit=%d, margin=%d)",
                req_id, requested_max, capped_max, input_token_count, state.context_limit, margin,
            )
            openai_body["max_completion_tokens"] = capped_max

    # Send main streaming request to backend (with auto-retry on context overflow)
    max_retries = 2
    response = None  # Track prepared StreamResponse across retries
    response_dump: _ResponseDumper | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = await send_to_backend(
                session,
                config,
                openai_body,
                req_id=req_id,
                backend_target=backend_target,
            )
        except Exception as e:
            log.error("[%s] Backend request failed: %s", req_id, e)
            if response is not None:
                await _emit_sse_error(response, "api_error", str(e), response_dump)
                if response_dump is not None:
                    response_dump.close()
                return response
            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": str(e)}},
                status=502,
            )

        if resp.status != 200:
            error_body = await resp.text()
            log.error("[%s] Backend returned %d: %s", req_id, resp.status, error_body[:500])

            retry = _context_overflow_retry_from_error(error_body, config, input_token_count)
            if retry is not None and attempt < max_retries:
                if retry.input_tokens is not None and retry.input_tokens > 0:
                    delta = retry.input_tokens - input_token_count
                    log.warning("[%s] Tokenizer delta: vLLM=%d, tiktoken=%d, delta=%d (%.1f%%)",
                                req_id, retry.input_tokens, input_token_count, delta,
                                (delta / input_token_count * 100) if input_token_count > 0 else 0)
                if retry.reason == "completion_limit":
                    log.warning("[%s] max_completion_tokens exceeds model limit: input≈%d, limit=%d → retrying with max_completion_tokens=%d (attempt %d)",
                                req_id, retry.input_tokens or 0, retry.ctx_limit, retry.max_completion_tokens, attempt + 1)
                else:
                    input_note = "at least " if retry.input_tokens_is_lower_bound else ""
                    log.warning("[%s] Context overflow on non-200: input=%s%s, limit=%d → retrying with max_completion_tokens=%d (attempt %d)",
                                req_id, input_note, retry.input_tokens, retry.ctx_limit, retry.max_completion_tokens, attempt + 1)
                resp.close()
                openai_body["max_completion_tokens"] = retry.max_completion_tokens
                continue

            if response is not None:
                await _emit_sse_error(response, "api_error", error_body, response_dump)
                if response_dump is not None:
                    response_dump.close()
                return response
            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": error_body}},
                status=resp.status,
            )

        log.info("[%s]     backend responded 200, streaming...", req_id)

        # Prepare the stream response immediately so the client gets 200 OK
        # headers without waiting for vLLM prefill. This prevents Claude Code's
        # internal retry logic from firing during long prefill latencies.
        # On context-overflow retry, reuse the already-prepared response
        # since only invisible SSE comments have been sent.
        if not use_image_agent:
            if response is None:
                response = await _prepare_stream_response(request)
                if response is None:
                    log.warning("[%s] Client disconnected before stream start", req_id)
                    resp.close()
                    return web.Response(status=499)
                response_dump = _ResponseDumper(req_id, config.dump_responses)

            try:
                await _send_keepalive(response, response_dump)
            except _ClientGone:
                log.warning("[%s] Client disconnected after prepare", req_id)
                resp.close()
                if response_dump is not None:
                    response_dump.close()
                return response

            # Emit message_start immediately so client sees input token count
            # and shows progress during prefill, instead of "initializing..."
            # until the first reasoning token arrives.
            if not getattr(response, "_relay_msg_start_sent", False):
                msg_start = make_anthropic_sse("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": f"msg_{os.urandom(12).hex()}",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": state.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": input_token_count,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "output_tokens": 0,
                        },
                    },
                })
                await response.write(msg_start)
                if response_dump is not None:
                    response_dump.write(msg_start)
                response._relay_msg_start_sent = True

            # Peek at SSE events with keepalive comments during waits.
            # This detects empty/error/context-overflow responses before
            # emitting any Anthropic message events to the client.
            sse_events = _debug_sse_stream(
                parse_sse_stream(resp),
                req_id,
                enabled=tool_debug_enabled,
                expected_tool_names=openai_tool_names,
            )
            try:
                buffered, has_content = await _peek_with_keepalive(
                    sse_events,
                    response,
                    response_dump,
                )
            except _ClientGone:
                log.warning("[%s] Client disconnected during peek", req_id)
                resp.close()
                if response_dump is not None:
                    response_dump.close()
                return response
            except Exception as exc:
                # Backend-read or parser error — not a client disconnect.
                log.error("[%s] Error peeking SSE stream: %s", req_id, exc)
                resp.close()
                await _emit_sse_error(response, "api_error", f"Backend stream error: {exc}", response_dump)
                if response_dump is not None:
                    response_dump.close()
                return response

            if not has_content:
                # Check if backend returned a context-overflow error
                error_msg = "Backend returned empty response"
                retry = None
                for ev in buffered:
                    if ev.data == "[DONE]":
                        continue
                    try:
                        ev_data = json.loads(ev.data)
                        if "error" in ev_data:
                            error_msg = ev_data["error"].get("message", str(ev_data["error"]))
                            retry = _context_overflow_retry_from_error(error_msg, config, input_token_count)
                            if retry is not None:
                                break
                    except (json.JSONDecodeError, TypeError):
                        pass

                if retry is not None and attempt < max_retries:
                    input_note = "at least " if retry.input_tokens_is_lower_bound else ""
                    log.warning("[%s] Context overflow: input=%s%s, limit=%d → retrying with max_completion_tokens=%d (attempt %d)",
                                req_id, input_note, retry.input_tokens, retry.ctx_limit,
                                retry.max_completion_tokens, attempt + 1)
                    resp.close()
                    openai_body["max_completion_tokens"] = retry.max_completion_tokens
                    # Only SSE comments have been sent so far — the client
                    # has received zero semantic events. We can transparently
                    # retry by re-entering the loop and reusing this same
                    # prepared StreamResponse (skip prepare on retry).
                    continue

                if attempt < max_retries:
                    log.warning(
                        "[%s] Backend returned no visible output/tool call; retrying with continuation instruction (attempt %d)",
                        req_id,
                        attempt + 1,
                    )
                    resp.close()
                    _append_empty_response_retry_instruction(openai_body)
                    continue

                log.error("[%s] EMPTY/ERROR response from backend — got %d events, error: %s",
                          req_id, len(buffered), error_msg)
                for i, ev in enumerate(buffered):
                    log.error("[%s]   event[%d]: %s", req_id, i, ev.data[:500])

                await _emit_sse_error(response, "overloaded_error", error_msg, response_dump)
                if response_dump is not None:
                    response_dump.close()
                return response

            async def _replay_and_continue():
                for ev in buffered:
                    yield ev
                async for ev in sse_events:
                    yield ev

            sse_source = _replay_and_continue()
        else:
            sse_source = None
            # Image agent path: prepare response the same way
            response = await _prepare_stream_response(request)
            if response is None:
                log.warning("[%s] Client disconnected before stream start", req_id)
                resp.close()
                return web.Response(status=499)
            response_dump = _ResponseDumper(req_id, config.dump_responses)
            try:
                await _send_keepalive(response, response_dump)
            except _ClientGone:
                log.warning("[%s] Client disconnected after prepare", req_id)
                resp.close()
                if response_dump is not None:
                    response_dump.close()
                return response

        break  # success

    bytes_sent = 0
    client_gone = False
    backend_error = False
    try:
        if use_image_agent:
            chunk_iter = image_agent_stream(
                resp,
                openai_body,
                session_id,
                session,
                config,
                request.app["image_cache"],
                req_id,
                emit_thinking=thinking_enabled,
                backend_target=backend_target,
            )
        else:
            chunk_iter = convert_openai_stream_to_anthropic(
                sse_source,
                req_id=req_id,
                input_tokens=input_token_count,
                emit_thinking=thinking_enabled,
                skip_message_start=getattr(response, "_relay_msg_start_sent", False),
            )
        async for chunk in chunk_iter:
            # Backend-read errors surface from the async for; client-write
            # errors surface from response.write(). Separate the scopes.
            try:
                await response.write(chunk)
                if response_dump is not None:
                    response_dump.write(chunk)
            except _DISCONNECT_EXC:
                client_gone = True
                log.warning("[%s] Client disconnected during streaming", req_id)
                break
            except _ClientGone:
                client_gone = True
                log.warning("[%s] Client disconnected during streaming", req_id)
                break
            bytes_sent += len(chunk)
    except _DISCONNECT_EXC as e:
        # This comes from the async iterator — it's a backend-read error,
        # not a client disconnect. The client is still connected.
        backend_error = True
        log.error("[%s] Backend stream read error: %s", req_id, e)
    except Exception as e:
        # Backend-read or converter error.
        backend_error = True
        log.error("[%s] Streaming error: %s", req_id, e, exc_info=True)

    log.info("[%s] <<< done, %d bytes sent", req_id, bytes_sent)
    if backend_error and not client_gone:
        await _emit_sse_error(response, "api_error", "Backend stream error", response_dump)
    elif not client_gone:
        try:
            await response.write_eof()
        except (RuntimeError, *_DISCONNECT_EXC) as e:
            log.warning("[%s] Client disconnected before EOF: %s", req_id, e)
    if response_dump is not None:
        response_dump.close()
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
        "backends": get_states_info(),
        "model_routes": {
            name: {
                "backend_url": route.backend_url,
                "upstream_model": route.upstream_model,
            }
            for name, route in config.model_routes.items()
        },
        "image_agent_enabled": config.image_agent_enabled,
        "vision_url": config.vision_url,
        "vision_model": config.vision_model,
    })


def _cleanup_debug_files(max_age_hours: int) -> int:
    """Delete debug dump files older than max_age_hours. Returns count deleted."""
    import time
    deleted = 0
    now = time.time()
    max_age_sec = max_age_hours * 3600

    if not os.path.isdir(DEBUG_DIR):
        return 0

    for fname in os.listdir(DEBUG_DIR):
        if not fname.endswith((".json", ".ndjson", ".sse")):
            continue
        fpath = os.path.join(DEBUG_DIR, fname)
        try:
            mtime = os.path.getmtime(fpath)
            if now - mtime > max_age_sec:
                os.remove(fpath)
                deleted += 1
        except OSError:
            pass  # race: file may have been deleted concurrently

    return deleted


async def on_startup(app: web.Application):
    app["session"] = aiohttp.ClientSession()
    config: ProxyConfig = app["config"]
    app["image_cache"] = ImageCache(config.image_cache_max_size, config.image_cache_ttl)

    if config.dump_requests or config.dump_responses or config.tool_debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    # Cleanup old debug files before starting
    if config.debug_max_age_hours is not None:
        deleted = _cleanup_debug_files(config.debug_max_age_hours)
        if deleted > 0:
            log.info("Cleaned up %d old debug file(s) from %s", deleted, DEBUG_DIR)

    for backend_url in config.backend_urls():
        await detect_backend(app["session"], backend_url)


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
