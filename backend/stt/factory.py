from __future__ import annotations

from pathlib import Path
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
            binary_path=settings.whispercpp_binary_path,
            model_path=Path(settings.whispercpp_model_path),
            language=language or settings.whispercpp_language,
            threads=settings.whispercpp_threads,
            step_ms=settings.whispercpp_step_ms,
            window_ms=settings.whispercpp_window_ms,
            overlap_ms=settings.whispercpp_overlap_ms,
            minimum_audio_ms=settings.whispercpp_minimum_audio_ms,
            timeout_seconds=settings.whispercpp_timeout_seconds,
            use_gpu=settings.whispercpp_use_gpu,
            flash_attention=settings.whispercpp_flash_attention,
        )
    raise ValueError(f"Unsupported STT provider: {settings.stt_provider}")
