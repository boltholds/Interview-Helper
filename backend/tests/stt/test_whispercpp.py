import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from stt.whispercpp import CommandResult, WhisperCppTranscriber


def _audio(milliseconds: int) -> bytes:
    return b"\x00\x00" * (16_000 * milliseconds // 1000)


@pytest.mark.asyncio
async def test_whispercpp_emits_partial_and_final_updates(tmp_path: Path) -> None:
    calls = 0

    async def runner(command: Sequence[str]) -> CommandResult:
        nonlocal calls
        calls += 1
        output_base = Path(command[command.index("-of") + 1])
        text = "worker retries" if calls == 1 else "worker retries and backoff"
        Path(f"{output_base}.json").write_text(
            json.dumps(
                {
                    "result": {"language": "en"},
                    "transcription": [{"text": text}],
                }
            ),
            encoding="utf-8",
        )
        return CommandResult(returncode=0, stdout="", stderr="")

    transcriber = WhisperCppTranscriber(
        binary_path="whisper-cli",
        model_path=tmp_path / "model.bin",
        step_ms=250,
        window_ms=1_000,
        overlap_ms=100,
        minimum_audio_ms=100,
        runner=runner,
    )

    partial = await transcriber.push_audio(_audio(300))
    final = await transcriber.flush()

    assert partial[0].is_final is False
    assert partial[0].text == "worker retries"
    assert final[0].is_final is True
    assert final[0].full_text == "worker retries and backoff"
    assert final[0].language == "en"


@pytest.mark.asyncio
async def test_whispercpp_finalizes_full_windows(tmp_path: Path) -> None:
    async def runner(command: Sequence[str]) -> CommandResult:
        output_base = Path(command[command.index("-of") + 1])
        Path(f"{output_base}.json").write_text(
            json.dumps({"transcription": [{"text": "final segment"}]}),
            encoding="utf-8",
        )
        return CommandResult(returncode=0, stdout="", stderr="")

    transcriber = WhisperCppTranscriber(
        binary_path="whisper-cli",
        model_path=tmp_path / "model.bin",
        step_ms=250,
        window_ms=1_000,
        overlap_ms=100,
        minimum_audio_ms=100,
        runner=runner,
    )

    updates = await transcriber.push_audio(_audio(1_000))

    assert len(updates) == 1
    assert updates[0].is_final is True
    assert updates[0].start_ms == 0
    assert updates[0].end_ms == 1_000
