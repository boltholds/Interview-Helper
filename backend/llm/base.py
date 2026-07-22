from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal


ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str


class LLMProvider(abc.ABC):
    name: str
    model: str

    @abc.abstractmethod
    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Yield text deltas for a chat completion."""
        raise NotImplementedError

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        chunks = [chunk async for chunk in self.stream(messages)]
        return "".join(chunks)

    @abc.abstractmethod
    async def close(self) -> None:
        """Release provider resources."""
        raise NotImplementedError
