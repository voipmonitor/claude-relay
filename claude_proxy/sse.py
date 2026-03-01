import json
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class SSEEvent:
    event: str | None = None
    data: str = ""


def make_anthropic_sse(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


async def parse_sse_stream(response) -> AsyncIterator[SSEEvent]:
    """Parse an aiohttp response as SSE events."""
    buffer = ""
    async for chunk in response.content.iter_any():
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            event = _parse_block(block)
            if event is not None:
                yield event
    # Handle trailing block without final \n\n
    if buffer.strip():
        event = _parse_block(buffer)
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
