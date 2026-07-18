from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ClientEventType(StrEnum):
    START_SESSION = "start_session"
    STOP_SESSION = "stop_session"
    TRANSCRIPT_CORRECTION = "transcript_correction"
    COMMIT_QUESTION = "commit_question"
    CANCEL_GENERATION = "cancel_generation"


class ServerEventType(StrEnum):
    SESSION_READY = "session_ready"
    EVENT_ACCEPTED = "event_accepted"
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    QUESTION_DETECTED = "question_detected"
    GENERATION_STARTED = "generation_started"
    ANSWER_DELTA = "answer_delta"
    ANSWER_COMPLETED = "answer_completed"
    ERROR = "error"


class ClientEvent(BaseModel):
    type: ClientEventType
    session_id: str
    sequence: int = Field(ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)
    sent_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ServerEvent(BaseModel):
    type: ServerEventType
    session_id: str
    sequence: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    sent_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
