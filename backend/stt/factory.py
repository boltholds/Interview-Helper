from __future__ import annotations

from typing import TYPE_CHECKING

from stt.base import StreamingTranscriber
from stt.stub import StubTranscriber
from stt.whispercpp import WhisperCppTranscriber

if TYPE_CHECKING:
    from app.core.config import Settings


def create_transcriber(settings: Settings, *, language: str | None = None) -> StreamingTranscriber:
    provider = settings.stt_provider.casefold()
    if provider == "stub":
        return StubTranscriber()
    if provider in {"whispercpp", "whisper.cpp"}:
        return WhisperCppTranscriber(
            server_url=settings.whispercpp_base_url,
            language=language or settings.whispercpp_language,
            step_ms=settings.whispercpp_step_ms,
            window_ms=settings.whispercpp_window_ms,
            overlap_ms=settings.whispercpp_overlap_ms,
            minimum_audio_ms=settings.whispercpp_minimum_audio_ms,
            timeout_seconds=settings.whispercpp_timeout_seconds,
        )
    raise ValueError(f"Unsupported STT provider: {settings.stt_provider}")
