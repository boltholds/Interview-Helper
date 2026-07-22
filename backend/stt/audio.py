from __future__ import annotations

import wave
from pathlib import Path

PCM16_SAMPLE_RATE = 16_000
PCM16_CHANNELS = 1
PCM16_SAMPLE_WIDTH = 2
BYTES_PER_MILLISECOND = PCM16_SAMPLE_RATE * PCM16_SAMPLE_WIDTH // 1000


def validate_pcm16(data: bytes) -> None:
    if len(data) % PCM16_SAMPLE_WIDTH != 0:
        raise ValueError("PCM16 audio payload must contain complete 16-bit samples")


def duration_ms(data: bytes) -> int:
    validate_pcm16(data)
    return len(data) // BYTES_PER_MILLISECOND


def milliseconds_to_bytes(milliseconds: int) -> int:
    if milliseconds < 0:
        raise ValueError("milliseconds cannot be negative")
    return milliseconds * BYTES_PER_MILLISECOND


def write_pcm16_wav(path: Path, data: bytes) -> None:
    validate_pcm16(data)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(PCM16_CHANNELS)
        output.setsampwidth(PCM16_SAMPLE_WIDTH)
        output.setframerate(PCM16_SAMPLE_RATE)
        output.writeframes(data)
