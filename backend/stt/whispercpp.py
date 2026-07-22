from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stt.audio import duration_ms, milliseconds_to_bytes, validate_pcm16, write_pcm16_wav
from stt.base import StreamingTranscriber, TranscriptUpdate, merge_transcript_text


class WhisperCppError(RuntimeError):
    """Raised when whisper.cpp cannot transcribe an audio window."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class WhisperResult:
    text: str
    language: str | None = None


CommandRunner = Callable[[Sequence[str]], Awaitable[CommandResult]]


async def _default_runner(command: Sequence[str]) -> CommandResult:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return CommandResult(
        returncode=process.returncode or 0,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


def _extract_json_text(payload: dict[str, Any]) -> WhisperResult:
    transcription = payload.get("transcription")
    if isinstance(transcription, list):
        parts = [
            str(segment.get("text", "")).strip()
            for segment in transcription
            if isinstance(segment, dict) and segment.get("text")
        ]
        text = " ".join(part for part in parts if part).strip()
    else:
        result = payload.get("result")
        text = str(result.get("text", "")).strip() if isinstance(result, dict) else ""
        if not text:
            text = str(payload.get("text", "")).strip()

    result_block = payload.get("result")
    language = None
    if isinstance(result_block, dict) and result_block.get("language"):
        language = str(result_block["language"])
    elif payload.get("language"):
        language = str(payload["language"])
    return WhisperResult(text=text, language=language)


class WhisperCppTranscriber(StreamingTranscriber):
    def __init__(
        self,
        *,
        binary_path: str,
        model_path: Path,
        language: str = "auto",
        threads: int = 4,
        step_ms: int = 3_000,
        window_ms: int = 12_000,
        overlap_ms: int = 1_000,
        minimum_audio_ms: int = 500,
        timeout_seconds: float = 120.0,
        use_gpu: bool = True,
        flash_attention: bool = True,
        runner: CommandRunner | None = None,
    ) -> None:
        if threads < 1:
            raise ValueError("threads must be positive")
        if step_ms < 250:
            raise ValueError("step_ms must be at least 250")
        if window_ms <= step_ms:
            raise ValueError("window_ms must be greater than step_ms")
        if overlap_ms < 0 or overlap_ms >= window_ms:
            raise ValueError("overlap_ms must be between 0 and window_ms")
        if minimum_audio_ms < 100:
            raise ValueError("minimum_audio_ms must be at least 100")

        self.binary_path = binary_path
        self.model_path = model_path
        self.language = language
        self.threads = threads
        self.step_ms = step_ms
        self.window_ms = window_ms
        self.overlap_ms = overlap_ms
        self.minimum_audio_ms = minimum_audio_ms
        self.timeout_seconds = timeout_seconds
        self.use_gpu = use_gpu
        self.flash_attention = flash_attention
        self._runner = runner or _default_runner
        self._uses_default_runner = runner is None

        self._pending = bytearray()
        self._committed_text = ""
        self._processed_ms = 0
        self._last_partial_size = 0
        self._last_partial_text = ""
        self._lock = asyncio.Lock()

    def validate_runtime(self) -> None:
        if not self._uses_default_runner:
            return
        executable = shutil.which(self.binary_path)
        if executable is None and not Path(self.binary_path).is_file():
            raise WhisperCppError(f"whisper-cli executable not found: {self.binary_path}")
        if not self.model_path.is_file():
            raise WhisperCppError(f"whisper.cpp model not found: {self.model_path}")

    def _command(self, wav_path: Path, output_base: Path) -> list[str]:
        command = [
            self.binary_path,
            "-m",
            str(self.model_path),
            "-f",
            str(wav_path),
            "-l",
            self.language,
            "-t",
            str(self.threads),
            "-oj",
            "-of",
            str(output_base),
            "-np",
            "--suppress-nst",
        ]
        if not self.use_gpu:
            command.append("-ng")
        if self.flash_attention:
            command.append("-fa")
        return command

    async def _transcribe(self, audio: bytes) -> WhisperResult:
        validate_pcm16(audio)
        if duration_ms(audio) < self.minimum_audio_ms:
            return WhisperResult(text="")
        self.validate_runtime()

        with tempfile.TemporaryDirectory(prefix="interview-helper-whisper-") as directory:
            workdir = Path(directory)
            wav_path = workdir / "audio.wav"
            output_base = workdir / "transcript"
            write_pcm16_wav(wav_path, audio)
            try:
                command_result = await asyncio.wait_for(
                    self._runner(self._command(wav_path, output_base)),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError as exc:
                raise WhisperCppError(
                    f"whisper.cpp timed out after {self.timeout_seconds:.0f} seconds"
                ) from exc

            if command_result.returncode != 0:
                detail = command_result.stderr.strip() or command_result.stdout.strip()
                raise WhisperCppError(
                    f"whisper.cpp exited with code {command_result.returncode}: {detail}"
                )

            json_path = Path(f"{output_base}.json")
            if not json_path.is_file():
                raise WhisperCppError(
                    "whisper.cpp did not create the expected JSON transcript"
                )
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise WhisperCppError(f"Cannot read whisper.cpp JSON output: {exc}") from exc
            if not isinstance(payload, dict):
                raise WhisperCppError("whisper.cpp JSON output must be an object")
            return _extract_json_text(payload)

    def _final_update(self, result: WhisperResult, start_ms: int, end_ms: int) -> TranscriptUpdate | None:
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
