"""Image agent: strip/cache images, intercept analyzeImage, call vision model."""

import json
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import AsyncIterator

import aiohttp

from .config import ProxyConfig
from .convert_request import _make_data_url
from .convert_stream import (
    StreamResult,
    convert_openai_stream_to_anthropic,
    _build_message_start,
    _build_message_delta,
    _make_message_id,
)
from .sse import SSEEvent, make_anthropic_sse, parse_sse_stream
from .backend import send_to_backend

log = logging.getLogger(__name__)

ANALYZE_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "analyzeImage",
        "description": (
            "View and analyze image(s). You MUST call this tool whenever "
            "the conversation contains [Image #N] references. "
            "Returns a detailed text description of the image content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "imageId": {
                    "type": "array",
                    "description": "Array of image IDs from [Image #N] placeholders in the conversation",
                    "items": {"type": "string"},
                },
                "task": {
                    "type": "string",
                    "description": (
                        "Specific analysis task based on the user's question. "
                        "Be detailed about what information you need from the image"
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Brief summary of the conversation context and what the user "
                        "is trying to accomplish, so the vision model understands the broader intent"
                    ),
                },
            },
            "required": ["imageId", "task"],
        },
    },
}

IMAGE_AGENT_SYSTEM_PROMPT = (
    "Images in this conversation are shown as [Image #N] placeholders. "
    "To see or analyze any image, you MUST call the analyzeImage tool with the correct imageId.\n"
    "When calling analyzeImage:\n"
    "- Pass the imageId(s) from the [Image #N] placeholders\n"
    "- Write a detailed task description explaining what information you need "
    "from the image based on the user's question\n"
    "- Include a context parameter summarizing the user's overall question and intent "
    "so the vision model understands what to focus on\n"
    "Never attempt to describe, interpret, or guess about image content without first "
    "calling analyzeImage. Always call the tool before responding about any image."
)

VISION_SYSTEM_PROMPT = (
    "Analyze the provided image(s) according to the task description below. "
    "Be thorough, specific, and accurate in your analysis. "
    "If conversation context is provided, use it to focus your analysis "
    "on the most relevant aspects of the image. Describe exactly what you observe."
)


class ImageCache:
    """Per-session LRU image cache with TTL."""

    def __init__(self, max_size: int = 100, ttl: int = 300):
        self._max_size = max_size
        self._ttl = ttl
        self._sessions: dict[str, OrderedDict[str, tuple[dict, float]]] = {}

    def store(self, session_id: str, image_key: str, source: dict):
        if session_id not in self._sessions:
            self._sessions[session_id] = OrderedDict()
        cache = self._sessions[session_id]
        cache[image_key] = (source, time.time())
        cache.move_to_end(image_key)
        # Evict oldest
        while len(cache) > self._max_size:
            cache.popitem(last=False)

    def get(self, session_id: str, image_key: str) -> dict | None:
        cache = self._sessions.get(session_id)
        if not cache or image_key not in cache:
            return None
        source, ts = cache[image_key]
        if time.time() - ts > self._ttl:
            del cache[image_key]
            return None
        cache.move_to_end(image_key)
        return source

    def cleanup_expired(self):
        now = time.time()
        empty = []
        for sid, cache in self._sessions.items():
            expired = [k for k, (_, ts) in cache.items() if now - ts > self._ttl]
            for k in expired:
                del cache[k]
            if not cache:
                empty.append(sid)
        for sid in empty:
            del self._sessions[sid]


_image_cache = ImageCache()


def has_images(body: dict) -> bool:
    """Check if the LAST user message contains image blocks.

    Only activates image agent for new images in the current turn,
    not for old images lingering in conversation history.
    """
    messages = body.get("messages", [])
    # Find the last user message
    last_user = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = msg
            break
    if last_user is None:
        return False

    content = last_user.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if block.get("type") == "image":
            return True
        # Images inside tool_result content arrays
        if block.get("type") == "tool_result" and isinstance(block.get("content"), list):
            for item in block["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    return True
    return False


def strip_and_cache_images(body: dict, session_id: str) -> dict:
    """Replace images with [Image #N] placeholders, cache originals, inject analyzeImage tool.

    Mutates and returns the body dict.
    """
    counter = 1

    for msg in body.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        new_content = []
        for block in content:
            if block.get("type") == "image" and block.get("source"):
                key = f"{session_id}_Image#{counter}"
                _image_cache.store(session_id, key, block["source"])
                new_content.append({
                    "type": "text",
                    "text": f'[Image #{counter}](call analyzeImage with imageId "{counter}" to view this image)',
                })
                counter += 1

            elif block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                new_items = []
                for item in block["content"]:
                    if isinstance(item, dict) and item.get("type") == "image" and item.get("source"):
                        key = f"{session_id}_Image#{counter}"
                        _image_cache.store(session_id, key, item["source"])
                        new_items.append({
                            "type": "text",
                            "text": f'[Image #{counter}](call analyzeImage with imageId "{counter}" to view this image)',
                        })
                        counter += 1
                    else:
                        new_items.append(item)
                block["content"] = new_items
                new_content.append(block)
            else:
                new_content.append(block)

        msg["content"] = new_content

    # Inject analyzeImage system prompt
    system = body.get("system", "")
    if isinstance(system, str):
        body["system"] = IMAGE_AGENT_SYSTEM_PROMPT + ("\n\n" + system if system else "")
    elif isinstance(system, list):
        body["system"] = [{"type": "text", "text": IMAGE_AGENT_SYSTEM_PROMPT}] + system

    # Inject analyzeImage tool (in Anthropic format — will be converted later)
    tools = body.setdefault("tools", [])
    # Check if already present
    if not any(t.get("name") == "analyzeImage" for t in tools):
        tools.append({
            "name": "analyzeImage",
            "description": ANALYZE_IMAGE_TOOL["function"]["description"],
            "input_schema": ANALYZE_IMAGE_TOOL["function"]["parameters"],
        })

    return body


@dataclass
class _InterceptState:
    """Tracks analyzeImage interception state while streaming."""
    intercepting: bool = False
    tool_id: str = ""
    tool_openai_index: int = -1
    arguments_buffer: str = ""
    max_block_index: int = -1


async def image_agent_stream(
    first_response: aiohttp.ClientResponse,
    openai_body: dict,
    session_id: str,
    session: aiohttp.ClientSession,
    config: ProxyConfig,
    req_id: str = "",
) -> AsyncIterator[bytes]:
    """Stream the first response, intercept analyzeImage tool_use, call vision, stitch follow-up.

    Works at the OpenAI SSE event level:
    1. Parse first response SSE events
    2. Convert to Anthropic SSE and yield (via convert_openai_stream_to_anthropic)
    3. If analyzeImage is detected: call vision model, build follow-up, stitch
    """
    _r = f"[{req_id}] " if req_id else ""

    # Phase 1: Stream first response, watch for analyzeImage at OpenAI SSE level
    intercept = _InterceptState()
    first_events = parse_sse_stream(first_response)

    result = StreamResult()
    block_index = 0

    async def _intercepting_events() -> AsyncIterator[SSEEvent]:
        """Yield OpenAI SSE events while tracking analyzeImage tool calls."""
        nonlocal intercept
        async for event in first_events:
            if event.data == "[DONE]":
                yield event
                break

            try:
                data = json.loads(event.data)
            except (json.JSONDecodeError, TypeError):
                yield event
                continue

            # Track tool calls
            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        name = func.get("name", "")
                        if name == "analyzeImage":
                            intercept.intercepting = True
                            intercept.tool_id = tc.get("id", "")
                            intercept.tool_openai_index = tc.get("index", 0)
                            log.info("%simage_agent: detected analyzeImage tool_call id=%s openai_idx=%d",
                                     _r, intercept.tool_id, intercept.tool_openai_index)
                        if intercept.intercepting and tc.get("index", 0) == intercept.tool_openai_index:
                            intercept.arguments_buffer += func.get("arguments", "")

            yield event

    # Emit message_start ourselves (we control the lifecycle)
    msg_id = _make_message_id()
    log.info("%simage_agent: Phase 1 START — emitting message_start id=%s", _r, msg_id)
    yield _build_message_start(msg_id, "proxy")

    # Stream first response converted to Anthropic format (skip wrapper — we handle it)
    async for chunk in convert_openai_stream_to_anthropic(
        _intercepting_events(),
        index_offset=0,
        skip_message_wrapper=True,
        result=result,
        req_id=req_id,
    ):
        yield chunk

    block_index = result.block_index
    log.info("%simage_agent: Phase 1 DONE — block_index=%d intercepting=%s stop_reason=%s",
             _r, block_index, intercept.intercepting, result.stop_reason)

    # If no analyzeImage was detected, emit message end and we're done
    if not intercept.intercepting:
        stop = result.stop_reason or "end_turn"
        log.info("%simage_agent: no analyzeImage — emitting message_delta(%s) + message_stop", _r, stop)
        yield _build_message_delta(stop, {})
        yield make_anthropic_sse("message_stop", {"type": "message_stop"})
        return

    log.info("%simage_agent: Phase 2 START — analyzeImage intercepted session=%s args=%s",
             _r, session_id, intercept.arguments_buffer[:300])

    # Phase 2: Parse analyzeImage arguments, call vision model
    try:
        args = json.loads(intercept.arguments_buffer)
    except json.JSONDecodeError:
        log.error("%simage_agent: FAILED to parse analyzeImage args: %s", _r, intercept.arguments_buffer)
        yield _build_message_delta("end_turn", {})
        yield make_anthropic_sse("message_stop", {"type": "message_stop"})
        return

    image_ids = args.get("imageId", [])
    task = args.get("task", "Describe this image in detail")
    context = args.get("context", "")
    log.info("%simage_agent: imageIds=%s task=%s context=%s", _r, image_ids, task[:200], context[:200])

    # Build vision request
    vision_content: list[dict] = []
    for img_id in image_ids:
        key = f"{session_id}_Image#{img_id}"
        source = _image_cache.get(session_id, key)
        if source:
            url = _make_data_url(source.get("data", ""), source.get("media_type", "image/png"))
            vision_content.append({
                "type": "image_url",
                "image_url": {"url": url},
            })
            log.info("%simage_agent: image %s found in cache, media_type=%s data_len=%d",
                     _r, key, source.get("media_type"), len(source.get("data", "")))
        else:
            log.warning("%simage_agent: image NOT FOUND in cache: %s", _r, key)

    # Add task text
    vision_prompt = f"Task: {task}"
    if context:
        vision_prompt += f"\nContext: {context}"
    vision_content.append({"type": "text", "text": vision_prompt})

    vision_body = {
        "model": config.vision_model,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": vision_content},
        ],
        "max_tokens": 4096,
        "stream": False,
    }

    log.info("%simage_agent: calling vision model=%s url=%s images=%d",
             _r, config.vision_model, config.vision_url, len(vision_content) - 1)

    try:
        async with session.post(
            config.vision_url,
            json=vision_body,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as vision_resp:
            log.info("%simage_agent: vision response status=%d", _r, vision_resp.status)
            vision_data = await vision_resp.json()
            if "error" in vision_data:
                err_msg = vision_data["error"].get("message", str(vision_data["error"]))
                log.error("%simage_agent: vision model ERROR: %s", _r, err_msg)
                vision_text = f"[Vision analysis error: {err_msg}]"
            else:
                vision_text = vision_data["choices"][0]["message"]["content"]
    except Exception as e:
        log.error("%simage_agent: vision model call FAILED: %s", _r, e, exc_info=True)
        vision_text = f"[Vision analysis failed: {e}]"

    log.info("%simage_agent: Phase 2 DONE — vision response %d chars: %s...", _r, len(vision_text), vision_text[:300])

    # Phase 3: Build follow-up request
    followup_body = dict(openai_body)
    followup_body["messages"] = list(openai_body["messages"])

    # Add assistant message with the analyzeImage tool call
    followup_body["messages"].append({
        "role": "assistant",
        "tool_calls": [{
            "id": intercept.tool_id,
            "type": "function",
            "function": {
                "name": "analyzeImage",
                "arguments": intercept.arguments_buffer,
            },
        }],
    })

    # Add tool result
    followup_body["messages"].append({
        "role": "tool",
        "content": vision_text,
        "tool_call_id": intercept.tool_id,
    })

    # Remove analyzeImage from tools to prevent recursive calls
    orig_tool_count = len(followup_body.get("tools", []))
    if "tools" in followup_body:
        followup_body["tools"] = [
            t for t in followup_body["tools"]
            if t.get("function", {}).get("name") != "analyzeImage"
        ]
        if not followup_body["tools"]:
            del followup_body["tools"]
    new_tool_count = len(followup_body.get("tools", []))
    log.info("%simage_agent: Phase 3 — follow-up built: %d msgs, tools %d→%d",
             _r, len(followup_body["messages"]), orig_tool_count, new_tool_count)

    # Phase 4: Send follow-up to text model
    log.info("%simage_agent: Phase 4 — sending follow-up to backend", _r)
    followup_resp = await send_to_backend(session, config, followup_body, req_id=req_id)
    log.info("%simage_agent: Phase 4 DONE — follow-up status=%d", _r, followup_resp.status)

    if followup_resp.status != 200:
        error_text = await followup_resp.text()
        log.error("%simage_agent: follow-up backend error %d: %s", _r, followup_resp.status, error_text[:500])
        yield _build_message_delta("end_turn", {})
        yield make_anthropic_sse("message_stop", {"type": "message_stop"})
        return

    # Phase 5: Stream follow-up with index_offset, skip message wrapper
    followup_events = parse_sse_stream(followup_resp)
    followup_result = StreamResult()

    log.info("%simage_agent: Phase 5 — streaming follow-up with index_offset=%d", _r, block_index)
    async for chunk in convert_openai_stream_to_anthropic(
        followup_events,
        index_offset=block_index,
        skip_message_wrapper=True,
        result=followup_result,
        req_id=req_id,
    ):
        yield chunk

    log.info("%simage_agent: Phase 5 DONE — follow-up block_index=%d stop_reason=%s",
             _r, followup_result.block_index, followup_result.stop_reason)

    # Phase 6: Send final message_delta + message_stop
    stop_reason = followup_result.stop_reason or "end_turn"
    log.info("%simage_agent: Phase 6 — emitting message_delta(%s) + message_stop", _r, stop_reason)
    yield _build_message_delta(stop_reason, {})
    yield make_anthropic_sse("message_stop", {"type": "message_stop"})
    log.info("%simage_agent: COMPLETE", _r)
