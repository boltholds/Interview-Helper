import asyncio

import httpx

from stt.base import TranscriptUpdate
from stt.whispercpp import WhisperCppTranscriber


def _audio(milliseconds: int) -> bytes:
    return b"\x00\x00" * (16_000 * milliseconds // 1000)


def test_whispercpp_emits_partial_and_final_updates() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path == "/inference"
        assert request.headers["content-type"].startswith("multipart/form-data")
        calls += 1
        text = "worker retries" if calls == 1 else "worker retries and backoff"
        return httpx.Response(200, json={"language": "en", "text": text})

    async def scenario() -> tuple[list[TranscriptUpdate], list[TranscriptUpdate]]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://whispercpp:8080",
        ) as client:
            transcriber = WhisperCppTranscriber(
                server_url="http://whispercpp:8080",
                step_ms=250,
                window_ms=1_000,
                overlap_ms=100,
                minimum_audio_ms=100,
                client=client,
            )
            await transcriber.validate_runtime()
            partial = await transcriber.push_audio(_audio(300))
            final = await transcriber.flush()
            await transcriber.close()
            return partial, final

    partial, final = asyncio.run(scenario())

    assert partial[0].is_final is False
    assert partial[0].text == "worker retries"
    assert final[0].is_final is True
    assert final[0].full_text == "worker retries and backoff"
    assert final[0].language == "en"


def test_whispercpp_finalizes_full_windows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"transcription": [{"text": "final segment"}]})

    async def scenario() -> list[TranscriptUpdate]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://whispercpp:8080",
        ) as client:
            transcriber = WhisperCppTranscriber(
                server_url="http://whispercpp:8080",
                step_ms=250,
                window_ms=1_000,
                overlap_ms=100,
                minimum_audio_ms=100,
                client=client,
            )
            updates = await transcriber.push_audio(_audio(1_000))
            await transcriber.close()
            return updates

    updates = asyncio.run(scenario())

    assert len(updates) == 1
    assert updates[0].is_final is True
    assert updates[0].start_ms == 0
    assert updates[0].end_ms == 1_000
