"""OpenAI SSE → Anthropic SSE streaming conversion."""

import json
import os
import logging
from typing import AsyncIterator

from .sse import SSEEvent, make_anthropic_sse

log = logging.getLogger(__name__)

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "stop_sequence",
}


def _make_message_id() -> str:
    return f"msg_{os.urandom(12).hex()}"


def _build_message_start(message_id: str, model: str, input_tokens: int = 0) -> bytes:
    return make_anthropic_sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        },
    })


def _build_message_delta(stop_reason: str, usage: dict | None) -> bytes:
    usage = usage or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", 0)
    return make_anthropic_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {
            "input_tokens": prompt_tokens - cached,
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_read_input_tokens": cached,
        },
    })


def _build_content_block_start(index: int, block: dict) -> bytes:
    return make_anthropic_sse("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": block,
    })


def _build_content_block_delta(index: int, delta: dict) -> bytes:
    return make_anthropic_sse("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": delta,
    })


def _build_content_block_stop(index: int) -> bytes:
    return make_anthropic_sse("content_block_stop", {
        "type": "content_block_stop",
        "index": index,
    })


class StreamResult:
    """Mutable container for the final block index after streaming completes."""

    def __init__(self):
        self.block_index: int = 0
        self.stop_reason: str = ""


async def convert_openai_stream_to_anthropic(
    sse_events: AsyncIterator[SSEEvent],
    index_offset: int = 0,
    skip_message_wrapper: bool = False,
    result: StreamResult | None = None,
    req_id: str = "",
    input_tokens: int = 0,
) -> AsyncIterator[bytes]:
    """Consume OpenAI SSE events and yield Anthropic SSE bytes.

    Args:
        sse_events: Async iterator of parsed SSE events from the OpenAI backend.
        index_offset: Offset to add to all content block indices (for follow-up stitching).
        skip_message_wrapper: If True, omit message_start and message_stop events.
        result: Optional mutable container; block_index is set after streaming ends.
        req_id: Request ID for log correlation.
        input_tokens: Input token count for message_start.
    """
    _r = f"[{req_id}] " if req_id else ""
    message_id = _make_message_id()
    model = "unknown"
    block_index = index_offset  # next block index to allocate
    current_index = -1  # currently open block index (-1 = none)
    thinking_started = False
    text_started = False
    tool_call_blocks: dict[int, int] = {}  # openai tool index → anthropic block index
    pending_usage: dict = {}
    pending_stop_reason = ""
    event_count = 0
    thinking_chars = 0
    text_chars = 0
    tools_finished = False  # set once finish_reason is received after tool calls
    closed_blocks: set[int] = set()  # anthropic block indices that have been stopped

    def _stop_block(idx: int) -> bytes:
        """Emit content_block_stop and track the closure."""
        closed_blocks.add(idx)
        return _build_content_block_stop(idx)

    log.info("%sconvert_stream: start index_offset=%d skip_wrapper=%s", _r, index_offset, skip_message_wrapper)

    async for event in sse_events:
        if event.data == "[DONE]":
            log.info("%sconvert_stream: [DONE] after %d events", _r, event_count)
            if event_count == 0:
                log.warning("%sconvert_stream: EMPTY response — backend sent [DONE] with no content!", _r)
            break

        event_count += 1
        # Log first few events verbatim for debugging
        if event_count <= 3:
            log.info("%sconvert_stream: event #%d raw: %s", _r, event_count, event.data[:500])

        try:
            data = json.loads(event.data)
        except (json.JSONDecodeError, TypeError):
            log.warning("%sconvert_stream: bad JSON event #%d: %s", _r, event_count, event.data[:200])
            continue

        # Extract model name from first chunk
        if "model" in data and model == "unknown":
            model = data["model"]
            log.info("%sconvert_stream: model=%s", _r, model)
            if not skip_message_wrapper:
                yield _build_message_start(message_id, model, input_tokens=input_tokens)

        choices = data.get("choices")
        if not choices:
            # usage-only chunk (some backends send this)
            if "usage" in data:
                pending_usage = data["usage"] or {}
                log.info("%sconvert_stream: usage-only chunk: %s", _r, pending_usage)
            continue

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Detect end-of-tools early so text gate opens for this same event
        if finish_reason and tool_call_blocks and finish_reason != "tool_calls":
            tools_finished = True

        # --- Thinking / reasoning_content ---
        reasoning = delta.get("reasoning_content") or ""
        if not reasoning:
            # Some backends use delta.thinking.content
            thinking_obj = delta.get("thinking")
            if isinstance(thinking_obj, dict):
                reasoning = thinking_obj.get("content", "")

        if reasoning:
            thinking_chars += len(reasoning)
            if not thinking_started:
                thinking_started = True
                current_index = block_index
                block_index += 1
                log.info("%sconvert_stream: thinking block START idx=%d", _r, current_index)
                yield _build_content_block_start(current_index, {
                    "type": "thinking",
                    "thinking": "",
                })
            yield _build_content_block_delta(current_index, {
                "type": "thinking_delta",
                "thinking": reasoning,
            })

        # --- Thinking signature ---
        thinking_obj = delta.get("thinking")
        if isinstance(thinking_obj, dict) and "signature" in thinking_obj:
            sig = thinking_obj["signature"]
            if sig and current_index >= 0 and thinking_started:
                log.info("%sconvert_stream: signature_delta len=%d", _r, len(sig))
                yield _build_content_block_delta(current_index, {
                    "type": "signature_delta",
                    "signature": sig,
                })

        # --- Text content ---
        text_content = delta.get("content")
        if text_content:
            text_chars += len(text_content)

            # Skip separator text between tool calls (OpenAI models emit "\n"
            # between tool_calls entries; the Anthropic protocol doesn't allow
            # text blocks interleaved with tool_use blocks).
            # Allow text through once tools are finished (non-tool finish_reason received).
            if not tool_call_blocks or tools_finished:
                # Close thinking block if still open
                if thinking_started and current_index >= 0 and not text_started:
                    log.info("%sconvert_stream: closing thinking block idx=%d (%d chars total)", _r, current_index, thinking_chars)
                    yield _stop_block(current_index)
                    thinking_started = False  # prevent double-close in tool_calls section

                # Close current tool_use block if transitioning back to text
                if tool_call_blocks and not text_started and current_index >= 0 and current_index not in closed_blocks:
                    log.info("%sconvert_stream: closing tool_use idx=%d for text-after-tools", _r, current_index)
                    yield _stop_block(current_index)

                if not text_started:
                    text_started = True
                    current_index = block_index
                    block_index += 1
                    log.info("%sconvert_stream: text block START idx=%d", _r, current_index)
                    yield _build_content_block_start(current_index, {
                        "type": "text",
                        "text": "",
                    })
                yield _build_content_block_delta(current_index, {
                    "type": "text_delta",
                    "text": text_content,
                })

        # --- Tool calls ---
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            # Close thinking if open and no text yet
            if thinking_started and not text_started and current_index >= 0:
                log.info("%sconvert_stream: closing thinking for tool_calls idx=%d", _r, current_index)
                yield _stop_block(current_index)
                thinking_started = False  # prevent double-close

            # Close text block if open
            if text_started:
                log.info("%sconvert_stream: closing text for tool_calls idx=%d (%d chars total)", _r, current_index, text_chars)
                yield _stop_block(current_index)
                text_started = False

            for tc in tool_calls:
                tc_index = tc.get("index", 0)
                func = tc.get("function", {})

                if tc_index not in tool_call_blocks:
                    # Close previous tool_use block before starting the next
                    if tool_call_blocks and current_index >= 0 and current_index not in closed_blocks:
                        log.info("%sconvert_stream: closing tool_use block idx=%d before next", _r, current_index)
                        yield _stop_block(current_index)

                    # New tool call — start block
                    tool_call_blocks[tc_index] = block_index
                    current_index = block_index
                    block_index += 1
                    tool_name = func.get("name", "")
                    tool_id = tc.get("id", "")
                    log.info("%sconvert_stream: tool_use START name=%s id=%s idx=%d", _r, tool_name, tool_id, current_index)
                    yield _build_content_block_start(current_index, {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": {},
                    })

                anthropic_idx = tool_call_blocks[tc_index]
                args_chunk = func.get("arguments", "")
                if args_chunk:
                    if anthropic_idx in closed_blocks:
                        log.warning("%sconvert_stream: late delta for closed block idx=%d (tc=%d), dropping",
                                    _r, anthropic_idx, tc_index)
                    else:
                        yield _build_content_block_delta(anthropic_idx, {
                            "type": "input_json_delta",
                            "partial_json": args_chunk,
                        })

        # --- Finish ---
        if finish_reason:
            pending_stop_reason = _FINISH_REASON_MAP.get(finish_reason, "end_turn")
            log.info("%sconvert_stream: finish_reason=%s → stop_reason=%s", _r, finish_reason, pending_stop_reason)

        # Usage from the chunk
        if "usage" in data:
            pending_usage = data["usage"] or {}

    # Close any open block
    if current_index >= 0 and current_index not in closed_blocks:
        log.info("%sconvert_stream: closing final block idx=%d", _r, current_index)
        yield _stop_block(current_index)

    log.info("%sconvert_stream: end events=%d thinking=%d text=%d tools=%d stop=%s",
             _r, event_count, thinking_chars, text_chars, len(tool_call_blocks), pending_stop_reason)

    if not skip_message_wrapper:
        if not pending_stop_reason:
            pending_stop_reason = "end_turn"
        yield _build_message_delta(pending_stop_reason, pending_usage)
        yield make_anthropic_sse("message_stop", {"type": "message_stop"})
        log.info("%sconvert_stream: emitted message_delta(%s) + message_stop", _r, pending_stop_reason)

    if result is not None:
        result.block_index = block_index
        result.stop_reason = pending_stop_reason
        log.info("%sconvert_stream: result set block_index=%d stop_reason=%s", _r, block_index, pending_stop_reason)
