from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal


ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Yield text deltas for a chat completion."""
        raise NotImplementedError

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        chunks = [chunk async for chunk in self.stream(messages)]
        return "".join(chunks)

    async def close(self) -> None:
        """Release provider resources."""
