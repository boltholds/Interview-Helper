from __future__ import annotations

import abc
import collections.abc
import dataclasses
import typing


ChatRole = typing.Literal["system", "user", "assistant"]


@dataclasses.dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str


class LLMProvider(abc.ABC):
    name: str
    model: str

    @abc.abstractmethod
    def stream(
        self,
        messages: collections.abc.Sequence[ChatMessage],
    ) -> collections.abc.AsyncIterator[str]:
        """Yield text deltas for a chat completion."""
        raise NotImplementedError

    async def complete(
        self,
        messages: collections.abc.Sequence[ChatMessage],
    ) -> str:
        chunks = [chunk async for chunk in self.stream(messages)]
        return "".join(chunks)

    @abc.abstractmethod
    async def close(self) -> None:
        """Release provider resources."""
        raise NotImplementedError
