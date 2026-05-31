import json
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class SSEEvent:
    event: str | None = None
    data: str = ""


def make_anthropic_sse(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


async def parse_sse_stream(response, *, max_buffer_bytes: int = 1 << 20) -> AsyncIterator[SSEEvent]:
    """Parse SSE stream with bounded buffer.

    Args:
        response: aiohttp ClientResponse
        max_buffer_bytes: Maximum buffer size before raising error (default 1MB)
    """
    buffer = bytearray()
    async for chunk in response.content.iter_any():
        buffer.extend(chunk)
        if len(buffer) > max_buffer_bytes:
            raise ValueError(f"SSE frame exceeded {max_buffer_bytes} bytes")
        while b"\n\n" in buffer:
            raw, _, rest = buffer.partition(b"\n\n")
            buffer = bytearray(rest)
            event = _parse_block(raw.decode("utf-8", errors="replace"))
            if event is not None:
                yield event
    # Handle trailing block without final \n\n
    if buffer:
        event = _parse_block(buffer.decode("utf-8", errors="replace"))
        if event is not None:
            yield event


def _parse_block(block: str) -> SSEEvent | None:
    event_type = None
    data_lines = []
    for line in block.split("\n"):
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
        elif line.startswith(":"):
            continue  # comment
    if not data_lines:
        return None
    return SSEEvent(event=event_type, data="\n".join(data_lines))
