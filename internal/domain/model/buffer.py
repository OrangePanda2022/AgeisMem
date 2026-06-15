"""用于存储最近的 N 条消息的环形缓冲区。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class DialogTurn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MessageBuffer:
    """环形缓冲区，记录最近 N 条对话轮次。"""

    def __init__(self, max_size: int = 20) -> None:
        self._buffer: deque[DialogTurn] = deque(maxlen=max_size)
        self._max_size = max_size

    def add(self, role: str, content: str) -> None:
        self._buffer.append(DialogTurn(role=role, content=content))

    def get_recent(self, n: Optional[int] = None) -> list[DialogTurn]:
        if n is None:
            n = self._max_size
        return list(self._buffer)[-n:]

    def get_all(self) -> list[DialogTurn]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    @property
    def size(self) -> int:
        return len(self._buffer)