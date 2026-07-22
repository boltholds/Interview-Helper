from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranscriptUpdate:
    text: str
    full_text: str
    is_final: bool
    start_ms: int
    end_ms: int
    language: str | None = None


class StreamingTranscriber(ABC):
    sample_rate: int = 16_000
    channels: int = 1
    sample_width: int = 2

    @abstractmethod
    async def push_audio(self, pcm16: bytes) -> list[TranscriptUpdate]:
        """Consume little-endian signed PCM16 mono audio."""
        raise NotImplementedError

    @abstractmethod
    async def flush(self) -> list[TranscriptUpdate]:
        """Finalize any buffered audio and return final transcript updates."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release provider resources."""


def merge_transcript_text(previous: str, incoming: str, *, max_overlap_words: int = 16) -> str:
    previous_words = previous.split()
    incoming_words = incoming.split()
    if not previous_words:
        return " ".join(incoming_words)
    if not incoming_words:
        return " ".join(previous_words)

    maximum = min(max_overlap_words, len(previous_words), len(incoming_words))
    previous_folded = [word.casefold().strip(".,!?;:—-()[]{}\"") for word in previous_words]
    incoming_folded = [word.casefold().strip(".,!?;:—-()[]{}\"") for word in incoming_words]
    overlap = 0
    for size in range(maximum, 0, -1):
        if previous_folded[-size:] == incoming_folded[:size]:
            overlap = size
            break
    return " ".join([*previous_words, *incoming_words[overlap:]]).strip()
