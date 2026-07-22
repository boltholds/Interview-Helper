from stt.base import StreamingTranscriber, TranscriptUpdate


class StubTranscriber(StreamingTranscriber):
    async def push_audio(self, pcm16: bytes) -> list[TranscriptUpdate]:
        return []

    async def flush(self) -> list[TranscriptUpdate]:
        return []
