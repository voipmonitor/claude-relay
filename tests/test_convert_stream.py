"""Golden tests for OpenAI → Anthropic stream conversion."""
import pytest
from claude_relay.convert_stream import (
    convert_openai_stream_to_anthropic,
    StreamResult,
)
from claude_relay.sse import SSEEvent


class MockSSEEvent:
    def __init__(self, data: str):
        self.data = data


async def fake_events(data_chunks: list[str]):
    """Helper to create mock SSE events."""
    for chunk in data_chunks:
        yield MockSSEEvent(chunk)


@pytest.mark.asyncio
async def test_glm_reasoning_field_maps_to_thinking_block_when_enabled():
    """GLM/vLLM streams reasoning as delta.reasoning, not reasoning_content."""
    events = fake_events([
        '{"model":"test"}',
        '{"choices":[{"delta":{"reasoning":"think"},"finish_reason":null}]}',
        '{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
        '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
        '[DONE]',
    ])
    result = StreamResult()
    chunks = [
        c async for c in convert_openai_stream_to_anthropic(
            events, result=result, emit_thinking=True,
        )
    ]
    output = b"".join(chunks).decode()

    assert '"type": "thinking"' in output
    assert '"thinking": "think"' in output
    assert output.count("event: content_block_start") == 2
    assert output.count("event: content_block_stop") == 2
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_reasoning_is_suppressed_when_thinking_not_enabled():
    """Do not emit Anthropic thinking blocks unless the client requested them."""
    events = fake_events([
        '{"model":"test"}',
        '{"choices":[{"delta":{"reasoning":"think"},"finish_reason":null}]}',
        '{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
        '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
        '[DONE]',
    ])
    result = StreamResult()
    chunks = [c async for c in convert_openai_stream_to_anthropic(events, result=result)]
    output = b"".join(chunks).decode()

    assert '"type": "thinking"' not in output
    assert '"thinking": "think"' not in output
    assert '"type": "text"' in output
    assert '"text": "Hello"' in output
    assert output.count("event: content_block_start") == 1
    assert output.count("event: content_block_stop") == 1
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_parallel_tool_calls_close_all_blocks():
    """Verify parallel tool calls all get content_block_stop."""
    events = fake_events([
        '{"model":"test"}',
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"foo","arguments":"{"}}]},"finish_reason":null}]}',
        '{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"t2","function":{"name":"bar","arguments":"{"}}]},"finish_reason":null}]}',
        '{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        '[DONE]',
    ])
    result = StreamResult()
    chunks = [c async for c in convert_openai_stream_to_anthropic(events, result=result)]
    output = b"".join(chunks).decode()

    assert output.count("event: content_block_start") == 2
    assert output.count("event: content_block_stop") == 2
    assert result.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_thinking_text_tool_interleave():
    """Verify thinking → text → tool_use sequence."""
    events = fake_events([
        '{"model":"test"}',
        '{"choices":[{"delta":{"reasoning_content":"thinking..."},"finish_reason":null}]}',
        '{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"search","arguments":"{"}}]},"finish_reason":null}]}',
        '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
        '[DONE]',
    ])
    result = StreamResult()
    chunks = [
        c async for c in convert_openai_stream_to_anthropic(
            events, result=result, emit_thinking=True,
        )
    ]
    output = b"".join(chunks).decode()

    # Should have: thinking block, text block, tool block
    assert "event: content_block_start" in output
    assert "event: content_block_stop" in output
    assert '"type": "thinking"' in output
    assert '"type": "text"' in output
    assert '"type": "tool_use"' in output


@pytest.mark.asyncio
async def test_malformed_sse_handling():
    """Verify malformed JSON events are skipped gracefully."""
    events = fake_events([
        '{"model":"test"}',
        'not valid json',
        '{"choices":[{"delta":{"content":"valid"},"finish_reason":null}]}',
        '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
        '[DONE]',
    ])
    result = StreamResult()
    chunks = [c async for c in convert_openai_stream_to_anthropic(events, result=result)]
    output = b"".join(chunks).decode()

    # Should have processed the valid chunk despite malformed one
    assert '"text": "valid"' in output
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_single_tool_call():
    """Verify single tool call is properly closed."""
    events = fake_events([
        '{"model":"test"}',
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"search","arguments":"{\\"query\\":\\"test\\"}"}}]},"finish_reason":null}]}',
        '{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        '[DONE]',
    ])
    result = StreamResult()
    chunks = [c async for c in convert_openai_stream_to_anthropic(events, result=result)]
    output = b"".join(chunks).decode()

    assert output.count("event: content_block_start") == 1
    assert output.count("event: content_block_stop") == 1
    assert result.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_tool_call_finish_without_tool_delta_downgrades_to_end_turn():
    """A backend must not make Claude Code expect a missing tool_use block."""
    events = fake_events([
        '{"model":"test"}',
        '{"choices":[{"delta":{"content":"I should call get_weather({\\"city\\":\\"Boston\\"})"},"finish_reason":"tool_calls"}]}',
        '[DONE]',
    ])
    result = StreamResult()
    chunks = [c async for c in convert_openai_stream_to_anthropic(events, result=result)]
    output = b"".join(chunks).decode()

    assert '"type": "tool_use"' not in output
    assert '"stop_reason": "end_turn"' in output
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_text_only_response_closes_block():
    """Text-only response closes its final text block (regression: was orphaned)."""
    events = fake_events([
        '{"model":"test"}',
        '{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
        '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
        '[DONE]',
    ])
    result = StreamResult()
    chunks = [c async for c in convert_openai_stream_to_anthropic(events, result=result)]
    output = b"".join(chunks).decode()

    assert output.count("event: content_block_start") == 1
    assert output.count("event: content_block_stop") == 1
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_thinking_then_text_closes_both():
    """Thinking + text response closes both blocks (regression: text was orphaned)."""
    events = fake_events([
        '{"model":"test"}',
        '{"choices":[{"delta":{"reasoning_content":"think"},"finish_reason":null}]}',
        '{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
        '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
        '[DONE]',
    ])
    result = StreamResult()
    chunks = [
        c async for c in convert_openai_stream_to_anthropic(
            events, result=result, emit_thinking=True,
        )
    ]
    output = b"".join(chunks).decode()

    assert output.count("event: content_block_start") == 2
    assert output.count("event: content_block_stop") == 2
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_thinking_only_closes_block():
    """Thinking-only response closes the thinking block at end-of-stream."""
    events = fake_events([
        '{"model":"test"}',
        '{"choices":[{"delta":{"reasoning_content":"think"},"finish_reason":null}]}',
        '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
        '[DONE]',
    ])
    result = StreamResult()
    chunks = [
        c async for c in convert_openai_stream_to_anthropic(
            events, result=result, emit_thinking=True,
        )
    ]
    output = b"".join(chunks).decode()

    assert output.count("event: content_block_start") == 1
    assert output.count("event: content_block_stop") == 1
    assert '"type": "thinking"' in output
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_skip_message_wrapper_balances_blocks():
    """skip_message_wrapper=True still closes all open blocks; wrapper events omitted."""
    events = fake_events([
        '{"model":"test"}',
        '{"choices":[{"delta":{"reasoning_content":"t"},"finish_reason":null}]}',
        '{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}',
        '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
        '[DONE]',
    ])
    result = StreamResult()
    chunks = [
        c async for c in convert_openai_stream_to_anthropic(
            events, index_offset=5, skip_message_wrapper=True, result=result,
            emit_thinking=True,
        )
    ]
    output = b"".join(chunks).decode()

    # Blocks balanced
    assert output.count("event: content_block_start") == output.count("event: content_block_stop") == 2
    # Wrapper events omitted
    assert "event: message_start" not in output
    assert "event: message_delta" not in output
    assert "event: message_stop" not in output
    # Index offset respected: first block allocated at 5, second at 6 → block_index ends at 7
    assert result.block_index == 7
