from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


class LLMProvider:
    name: str
    model: str

    def stream(self, messages: list[ChatMessage]):
        """Yield text deltas for a chat completion."""
        raise NotImplementedError

    async def complete(self, messages: list[ChatMessage]) -> str:
        chunks = [chunk async for chunk in self.stream(messages)]
        return "".join(chunks)

    async def close(self) -> None:
        """Release provider resources."""
        return None
