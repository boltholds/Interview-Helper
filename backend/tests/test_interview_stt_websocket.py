from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.interview.answering import AnswerContext
from app.main import app
from llm.base import ChatMessage
from stt.base import StreamingTranscriber, TranscriptUpdate


def test_interview_websocket_accepts_pcm16_after_start(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.ws.interview.get_settings",
        lambda: Settings(stt_provider="stub"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws/interview/session-1") as websocket:
        ready = websocket.receive_json()
        assert ready["type"] == "session_ready"
        assert ready["payload"]["audio"]["sample_rate"] == 16_000

        websocket.send_json(
            {
                "type": "start_session",
                "session_id": "session-1",
                "sequence": 0,
                "payload": {"language": "auto"},
            }
        )
        assert websocket.receive_json()["type"] == "event_accepted"
        assert websocket.receive_json()["type"] == "stt_ready"

        websocket.send_bytes(b"\x00\x00" * 1_600)
        websocket.send_json(
            {
                "type": "stop_session",
                "session_id": "session-1",
                "sequence": 1,
                "payload": {},
            }
        )
        assert websocket.receive_json()["type"] == "event_accepted"
        assert websocket.receive_json()["type"] == "stt_stopped"


def test_dual_source_audio_routes_to_separate_transcribers(monkeypatch) -> None:
    instances = []

    class FakeTranscriber(StreamingTranscriber):
        def __init__(self) -> None:
            self.audio = []
            instances.append(self)

        async def push_audio(self, pcm16: bytes) -> list[TranscriptUpdate]:
            self.audio.append(pcm16)
            label = (
                "Interviewer statement"
                if len(instances) == 2 and self is instances[0]
                else "Candidate answers"
            )
            return [TranscriptUpdate(label, label, True, 0, 100, "en")]

        async def flush(self) -> list[TranscriptUpdate]:
            return []

        async def close(self) -> None:
            return None

    monkeypatch.setattr("app.ws.interview.get_settings", lambda: Settings(stt_provider="stub"))
    monkeypatch.setattr(
        "app.ws.interview.create_transcriber", lambda *args, **kwargs: FakeTranscriber()
    )
    client = TestClient(app)

    with client.websocket_connect("/ws/interview/dual") as websocket:
        ready = websocket.receive_json()
        assert ready["payload"]["audio"]["source_tags"] == {"interviewer": 1, "candidate": 2}
        websocket.send_json(
            {
                "type": "start_session",
                "session_id": "dual",
                "sequence": 0,
                "payload": {"audio_protocol": "source_tagged_pcm_v1"},
            }
        )
        websocket.receive_json()
        stt_ready = websocket.receive_json()
        assert stt_ready["payload"]["sources"] == ["interviewer", "candidate"]

        websocket.send_bytes(b"\x01\x00\x00")
        interviewer = websocket.receive_json()
        assert interviewer["payload"]["source"] == "interviewer"
        assert interviewer["payload"]["speaker"] == "interviewer"
        websocket.send_bytes(b"\x02\x01\x00")
        candidate = websocket.receive_json()
        assert candidate["payload"]["source"] == "candidate"
        assert "interviewer: Interviewer statement" in candidate["payload"]["timeline_text"]
        assert "candidate: Candidate answers" in candidate["payload"]["timeline_text"]
        assert instances[0].audio == [b"\x00\x00"]
        assert instances[1].audio == [b"\x01\x00"]


def test_stop_flushes_sources_by_audio_arrival_order(monkeypatch) -> None:
    instances = []

    class FakeTranscriber(StreamingTranscriber):
        def __init__(self) -> None:
            self.index = len(instances)
            instances.append(self)

        async def push_audio(self, pcm16: bytes) -> list[TranscriptUpdate]:
            return []

        async def flush(self) -> list[TranscriptUpdate]:
            text = "interviewer residual" if self.index == 0 else "candidate residual"
            return [TranscriptUpdate(text, text, True, 0, 100, "en")]

        async def close(self) -> None:
            return None

    monkeypatch.setattr("app.ws.interview.get_settings", lambda: Settings(stt_provider="stub"))
    monkeypatch.setattr(
        "app.ws.interview.create_transcriber", lambda *args, **kwargs: FakeTranscriber()
    )
    client = TestClient(app)

    with client.websocket_connect("/ws/interview/flush-order") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "start_session",
                "session_id": "flush-order",
                "sequence": 0,
                "payload": {"audio_protocol": "source_tagged_pcm_v1"},
            }
        )
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_bytes(b"\x02\x00\x00")
        websocket.send_bytes(b"\x01\x00\x00")
        # A later candidate frame must not move its residual audio behind the
        # interviewer stream: ordering uses the first pending frame, not the last.
        websocket.send_bytes(b"\x02\x01\x00")
        websocket.send_json(
            {
                "type": "stop_session",
                "session_id": "flush-order",
                "sequence": 1,
                "payload": {},
            }
        )

        assert websocket.receive_json()["type"] == "event_accepted"
        first = websocket.receive_json()
        second = websocket.receive_json()
        assert first["payload"]["source"] == "candidate"
        assert second["payload"]["source"] == "interviewer"
        assert websocket.receive_json()["type"] == "stt_stopped"


def test_committed_question_streams_answer(monkeypatch) -> None:
    class FakeProvider:
        async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
            assert messages[-1].role == "user"
            yield "Первый тезис. "
            yield "Готовый ответ."

        async def close(self) -> None:
            return None

    async def fake_prepare_answer_context(*args, **kwargs) -> AnswerContext:
        assert kwargs["question"] == "Как работает event loop?"
        return AnswerContext(
            messages=[ChatMessage(role="user", content=kwargs["question"])],
            sources=[{"title": "AsyncIO", "source_path": "asyncio.md", "score": 0.9}],
        )

    monkeypatch.setattr(
        "app.ws.interview.get_settings",
        lambda: Settings(stt_provider="stub", openrouter_api_key="test-key"),
    )
    monkeypatch.setattr(
        "app.ws.interview.prepare_answer_context",
        fake_prepare_answer_context,
    )
    monkeypatch.setattr(
        "app.ws.interview.create_llm_provider",
        lambda settings: FakeProvider(),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws/interview/session-2") as websocket:
        assert websocket.receive_json()["type"] == "session_ready"
        websocket.send_json(
            {
                "type": "start_session",
                "session_id": "session-2",
                "sequence": 0,
                "payload": {"language": "ru", "role": "Python Developer"},
            }
        )
        assert websocket.receive_json()["type"] == "event_accepted"
        assert websocket.receive_json()["type"] == "stt_ready"

        websocket.send_json(
            {
                "type": "commit_question",
                "session_id": "session-2",
                "sequence": 1,
                "payload": {"question": "Как работает event loop?"},
            }
        )
        assert websocket.receive_json()["type"] == "event_accepted"
        assert websocket.receive_json()["type"] == "question_detected"
        started = websocket.receive_json()
        assert started["type"] == "generation_started"
        assert started["payload"]["sources"][0]["title"] == "AsyncIO"
        first_delta = websocket.receive_json()
        second_delta = websocket.receive_json()
        completed = websocket.receive_json()
        assert first_delta["type"] == "answer_delta"
        assert second_delta["payload"]["text"] == "Первый тезис. Готовый ответ."
        assert completed["type"] == "answer_completed"
        assert completed["payload"]["answer"] == "Первый тезис. Готовый ответ."
