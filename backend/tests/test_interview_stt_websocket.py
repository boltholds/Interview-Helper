from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app


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
