from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class SSEMessage:
    event: str | None
    data: str


async def iter_sse_messages(
    byte_iter: AsyncIterator[bytes],
) -> AsyncIterator[SSEMessage]:
    buffer = ""
    async for chunk in byte_iter:
        buffer += chunk.decode("utf-8", errors="replace")
        buffer = buffer.replace("\r\n", "\n")
        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            if not raw_event.strip():
                continue
            event_name = None
            data_lines: list[str] = []
            for line in raw_event.split("\n"):
                if line.startswith("event:"):
                    event_name = line[len("event:") :].strip() or None
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].lstrip())
            if data_lines:
                yield SSEMessage(event=event_name, data="\n".join(data_lines))


def format_sse(data: str, event: str | None = None) -> str:
    if event:
        return f"event: {event}\ndata: {data}\n\n"
    return f"data: {data}\n\n"
