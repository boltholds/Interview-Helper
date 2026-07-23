from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.interview.answering import AnswerContext
from app.main import app
from llm.base import ChatMessage


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
