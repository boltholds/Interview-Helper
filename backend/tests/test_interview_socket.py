from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_interview_socket_accepts_valid_event() -> None:
    with client.websocket_connect("/ws/interview/session-1") as websocket:
        ready = websocket.receive_json()
        assert ready["type"] == "session_ready"
        assert ready["session_id"] == "session-1"

        websocket.send_json(
            {
                "type": "start_session",
                "session_id": "session-1",
                "sequence": 1,
                "payload": {"target_role": "Python Developer"},
            }
        )

        accepted = websocket.receive_json()
        assert accepted["type"] == "event_accepted"
        assert accepted["sequence"] == 1
        assert accepted["payload"]["accepted_type"] == "start_session"


def test_interview_socket_reports_invalid_event() -> None:
    with client.websocket_connect("/ws/interview/session-2") as websocket:
        websocket.receive_json()
        websocket.send_text("not-json")

        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["payload"]["code"] == "invalid_event"
