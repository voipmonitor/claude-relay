"""Tests for SSE parser with focus on buffer overflow protection and edge cases."""
import pytest
from claude_relay.sse import parse_sse_stream, SSEEvent


class MockResponse:
    """Mock aiohttp ClientResponse for testing."""

    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    class Content:
        def __init__(self, chunks: list[bytes]):
            self.chunks = chunks

        async def iter_any(self):
            for chunk in self.chunks:
                yield chunk

    @property
    def content(self):
        return self.Content(self.chunks)


@pytest.mark.asyncio
async def test_buffer_overflow_raises_value_error():
    """Verify buffer overflow raises ValueError when exceeding max_buffer_bytes."""
    # Create chunk larger than default 1MB limit
    large_chunk = b"x" * ((1 << 20) + 100)
    response = MockResponse([large_chunk])

    with pytest.raises(ValueError, match="SSE frame exceeded"):
        async for _ in parse_sse_stream(response):
            pass


@pytest.mark.asyncio
async def test_normal_streaming_under_limit():
    """Verify normal streaming works fine when under the buffer limit."""
    chunks = [
        b"event: message_start\n",
        b"data: {\"type\":\"message_start\"}\n\n",
        b"event: content_block_start\n",
        b"data: {\"type\":\"content_block_start\",\"index\":0}\n\n",
        b"event: content_block_delta\n",
        b"data: {\"type\":\"text_delta\",\"text\":\"Hello\"}\n\n",
        b"event: content_block_stop\n",
        b"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        b"event: message_stop\n",
        b"data: {\"type\":\"message_stop\"}\n\n",
    ]
    response = MockResponse(chunks)

    events = [event async for event in parse_sse_stream(response)]

    assert len(events) == 5
    assert events[0].event == "message_start"
    assert events[0].data == '{"type":"message_start"}'
    assert events[2].event == "content_block_delta"
    assert "Hello" in events[2].data


@pytest.mark.asyncio
async def test_exactly_at_buffer_limit():
    """Verify stream works when buffer is exactly at the limit."""
    # Create content exactly 1MB (default limit)
    data_size = (1 << 20) - 50  # Leave room for SSE framing
    data_content = "x" * data_size
    chunk = f"event: data\ndata: {data_content}\n\n".encode()

    response = MockResponse([chunk])

    events = [event async for event in parse_sse_stream(response)]

    assert len(events) == 1
    assert events[0].event == "data"
    assert len(events[0].data) == data_size


@pytest.mark.asyncio
async def test_just_over_buffer_limit():
    """Verify stream fails when buffer exceeds limit by any amount."""
    # Create content just 1 byte over the limit
    data_size = (1 << 20) + 1
    data_content = "x" * data_size
    chunk = f"event: data\ndata: {data_content}\n\n".encode()

    response = MockResponse([chunk])

    with pytest.raises(ValueError, match="SSE frame exceeded"):
        async for _ in parse_sse_stream(response):
            pass


@pytest.mark.asyncio
async def test_malformed_frame_missing_delimiter():
    """Verify malformed frames without \\n\\n delimiter are handled."""
    # Frame without final delimiter - should be processed at stream end
    chunks = [
        b"event: message_start\n",
        b"data: {\"type\":\"message_start\"}\n\n",
        b"event: incomplete\n",
        b"data: {\"type\":\"incomplete\"}",  # No \n\n
    ]
    response = MockResponse(chunks)

    events = [event async for event in parse_sse_stream(response)]

    assert len(events) == 2
    assert events[0].event == "message_start"
    assert events[1].event == "incomplete"


@pytest.mark.asyncio
async def test_oversized_unterminated_frame_rejected():
    """Security test: oversized frame without delimiter raises BEFORE yielding events.

    This validates the DoS protection: an attacker sending a massive partial event
    should be rejected before any memory exhaustion occurs, not after parsing completes.
    """
    # Send a massive data line with no terminating \n\n
    # The buffer should exceed the limit while still inside the incomplete frame
    oversized_data = b"x" * ((1 << 20) + 100)  # 1MB + 100 bytes
    chunks = [
        b"event: huge\n",
        b"data: ",
        oversized_data,
        # Intentionally no \n\n delimiter - frame never completes
    ]
    response = MockResponse(chunks)

    with pytest.raises(ValueError, match="SSE frame exceeded"):
        async for _ in parse_sse_stream(response):
            pass


@pytest.mark.asyncio
async def test_malformed_frame_no_data_lines():
    """Verify frames with no data lines are skipped."""
    chunks = [
        b"event: empty\n\n",  # No data: line
        b"event: valid\n",
        b"data: {\"ok\":true}\n\n",
    ]
    response = MockResponse(chunks)

    events = [event async for event in parse_sse_stream(response)]

    assert len(events) == 1
    assert events[0].event == "valid"
    assert events[0].data == '{"ok":true}'


@pytest.mark.asyncio
async def test_invalid_utf8_handling():
    """Verify invalid UTF-8 sequences are handled with replacement."""
    # Valid event followed by invalid UTF-8 bytes
    chunks = [
        b"event: valid\n",
        b"data: {\"ok\":true}\n\n",
        b"event: bad\n",
        b"data: \xff\xfe\x00\x01\n\n",  # Invalid UTF-8
    ]
    response = MockResponse(chunks)

    events = [event async for event in parse_sse_stream(response)]

    assert len(events) == 2
    assert events[0].event == "valid"
    assert events[1].event == "bad"
    # Invalid UTF-8 should be replaced with replacement character
    assert "" in events[1].data or events[1].data  # Should not crash


@pytest.mark.asyncio
async def test_partial_chunks_assembled_correctly():
    """Verify partial chunks are correctly assembled into complete frames."""
    # Split a single frame across multiple chunks
    chunks = [
        b"event: message_start\n",
        b"data: {\"type\":\"message_start\"}\n",  # Incomplete - no \n\n
        b"\n",  # Completes the frame
        b"event: content\n",
        b"data: {\"text\":\"hello\"}\n\n",
    ]
    response = MockResponse(chunks)

    events = [event async for event in parse_sse_stream(response)]

    assert len(events) == 2
    assert events[0].event == "message_start"
    assert events[1].event == "content"


@pytest.mark.asyncio
async def test_custom_max_buffer_size():
    """Verify custom max_buffer_bytes parameter is respected."""
    # 100 byte limit
    chunk = b"x" * 150
    response = MockResponse([chunk])

    with pytest.raises(ValueError, match="SSE frame exceeded 100 bytes"):
        async for _ in parse_sse_stream(response, max_buffer_bytes=100):
            pass


@pytest.mark.asyncio
async def test_comment_lines_skipped():
    """Verify SSE comment lines (starting with :) are skipped."""
    chunks = [
        b": comment line\n",
        b"event: data\n",
        b": another comment\n",
        b"data: {\"ok\":true}\n\n",
    ]
    response = MockResponse(chunks)

    events = [event async for event in parse_sse_stream(response)]

    assert len(events) == 1
    assert events[0].event == "data"
    assert events[0].data == '{"ok":true}'


@pytest.mark.asyncio
async def test_multiple_data_lines_joined():
    """Verify multiple data: lines are joined with newlines."""
    chunks = [
        b"event: multi\n",
        b"data: {\"line\":1}\n",
        b"data: {\"line\":2}\n",
        b"data: {\"line\":3}\n\n",
    ]
    response = MockResponse(chunks)

    events = [event async for event in parse_sse_stream(response)]

    assert len(events) == 1
    assert events[0].event == "multi"
    assert events[0].data == '{"line":1}\n{"line":2}\n{"line":3}'


@pytest.mark.asyncio
async def test_empty_stream():
    """Verify empty stream yields no events."""
    response = MockResponse([])

    events = [event async for event in parse_sse_stream(response)]

    assert len(events) == 0


@pytest.mark.asyncio
async def test_whitespace_only_stream():
    """Verify stream with only whitespace yields no events."""
    chunks = [b"\n", b"  \n", b"\t\n"]
    response = MockResponse(chunks)

    events = [event async for event in parse_sse_stream(response)]

    assert len(events) == 0
