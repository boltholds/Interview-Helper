from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import dataclass
from typing import Any

import httpx

from stt.audio import duration_ms, milliseconds_to_bytes, validate_pcm16
from stt.base import StreamingTranscriber, TranscriptUpdate, merge_transcript_text


class WhisperCppError(RuntimeError):
    """Raised when the whisper.cpp service cannot transcribe an audio window."""


@dataclass(frozen=True, slots=True)
class WhisperResult:
    text: str
    language: str | None = None


def _extract_segment_text(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        str(segment.get("text", "")).strip()
        for segment in value
        if isinstance(segment, dict) and segment.get("text")
    ]


def _extract_json_text(payload: dict[str, Any]) -> WhisperResult:
    parts = _extract_segment_text(payload.get("transcription"))
    if not parts:
        parts = _extract_segment_text(payload.get("segments"))

    result_block = payload.get("result")
    text = " ".join(part for part in parts if part).strip()
    if not text and isinstance(result_block, dict):
        text = str(result_block.get("text", "")).strip()
    if not text:
        text = str(payload.get("text", "")).strip()

    language = None
    if isinstance(result_block, dict) and result_block.get("language"):
        language = str(result_block["language"])
    elif payload.get("language"):
        language = str(payload["language"])
    return WhisperResult(text=text, language=language)


def _wav_bytes(pcm16: bytes) -> bytes:
    validate_pcm16(pcm16)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(pcm16)
    return output.getvalue()


class WhisperCppTranscriber(StreamingTranscriber):
    def __init__(
        self,
        *,
        server_url: str,
        language: str = "auto",
        step_ms: int = 3_000,
        window_ms: int = 12_000,
        overlap_ms: int = 1_000,
        minimum_audio_ms: int = 500,
        timeout_seconds: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if step_ms < 250:
            raise ValueError("step_ms must be at least 250")
        if window_ms <= step_ms:
            raise ValueError("window_ms must be greater than step_ms")
        if overlap_ms < 0 or overlap_ms >= window_ms:
            raise ValueError("overlap_ms must be between 0 and window_ms")
        if minimum_audio_ms < 100:
            raise ValueError("minimum_audio_ms must be at least 100")
        if not server_url.strip():
            raise ValueError("whisper.cpp server URL cannot be empty")

        self.server_url = server_url.rstrip("/")
        self.language = language
        self.step_ms = step_ms
        self.window_ms = window_ms
        self.overlap_ms = overlap_ms
        self.minimum_audio_ms = minimum_audio_ms
        self.timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.server_url,
            timeout=httpx.Timeout(timeout_seconds),
        )

        self._pending = bytearray()
        self._committed_text = ""
        self._processed_ms = 0
        self._last_partial_size = 0
        self._last_partial_text = ""
        self._lock = asyncio.Lock()

    async def validate_runtime(self) -> None:
        try:
            response = await self._client.get("/health")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WhisperCppError(
                f"whisper.cpp service is unavailable at {self.server_url}: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise WhisperCppError("whisper.cpp health response is not valid JSON") from exc
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise WhisperCppError(f"whisper.cpp service is not ready: {payload}")

    async def _transcribe(self, audio: bytes) -> WhisperResult:
        validate_pcm16(audio)
        if duration_ms(audio) < self.minimum_audio_ms:
            return WhisperResult(text="")

        try:
            response = await self._client.post(
                "/inference",
                files={"file": ("audio.wav", _wav_bytes(audio), "audio/wav")},
                data={
                    "language": self.language,
                    "response_format": "verbose_json",
                    "temperature": "0.0",
                    "temperature_inc": "0.2",
                    "no_speech_thold": "0.6",
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            raise WhisperCppError(
                f"whisper.cpp inference failed with HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise WhisperCppError(f"whisper.cpp inference request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise WhisperCppError("whisper.cpp inference response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise WhisperCppError("whisper.cpp inference response must be a JSON object")
        return _extract_json_text(payload)

    def _final_update(
        self,
        result: WhisperResult,
        start_ms: int,
        end_ms: int,
    ) -> TranscriptUpdate | None:
        if not result.text:
            return None
        previous = self._committed_text
        merged = merge_transcript_text(previous, result.text)
        delta = merged[len(previous):].strip() if merged.startswith(previous) else result.text
        self._committed_text = merged
        self._last_partial_text = ""
        return TranscriptUpdate(
            text=delta or result.text,
            full_text=merged,
            is_final=True,
            start_ms=start_ms,
            end_ms=end_ms,
            language=result.language,
        )

    async def push_audio(self, pcm16: bytes) -> list[TranscriptUpdate]:
        validate_pcm16(pcm16)
        if not pcm16:
            return []

        async with self._lock:
            self._pending.extend(pcm16)
            updates: list[TranscriptUpdate] = []
            window_bytes = milliseconds_to_bytes(self.window_ms)
            consumed_bytes = milliseconds_to_bytes(self.window_ms - self.overlap_ms)

            while len(self._pending) >= window_bytes:
                audio_window = bytes(self._pending[:window_bytes])
                result = await self._transcribe(audio_window)
                update = self._final_update(
                    result,
                    self._processed_ms,
                    self._processed_ms + self.window_ms,
                )
                if update is not None:
                    updates.append(update)
                del self._pending[:consumed_bytes]
                self._processed_ms += self.window_ms - self.overlap_ms
                self._last_partial_size = 0

            step_bytes = milliseconds_to_bytes(self.step_ms)
            if (
                len(self._pending) >= milliseconds_to_bytes(self.minimum_audio_ms)
                and len(self._pending) - self._last_partial_size >= step_bytes
            ):
                result = await self._transcribe(bytes(self._pending))
                self._last_partial_size = len(self._pending)
                if result.text and result.text != self._last_partial_text:
                    self._last_partial_text = result.text
                    updates.append(
                        TranscriptUpdate(
                            text=result.text,
                            full_text=merge_transcript_text(
                                self._committed_text,
                                result.text,
                            ),
                            is_final=False,
                            start_ms=self._processed_ms,
                            end_ms=self._processed_ms + duration_ms(self._pending),
                            language=result.language,
                        )
                    )
            return updates

    async def flush(self) -> list[TranscriptUpdate]:
        async with self._lock:
            if duration_ms(self._pending) < self.minimum_audio_ms:
                self._pending.clear()
                self._last_partial_size = 0
                self._last_partial_text = ""
                return []

            end_ms = self._processed_ms + duration_ms(self._pending)
            result = await self._transcribe(bytes(self._pending))
            update = self._final_update(result, self._processed_ms, end_ms)
            self._pending.clear()
            self._processed_ms = end_ms
            self._last_partial_size = 0
            return [update] if update is not None else []

    async def close(self) -> None:
        self._pending.clear()
        self._last_partial_size = 0
        self._last_partial_text = ""
        if self._owns_client:
            await self._client.aclose()
