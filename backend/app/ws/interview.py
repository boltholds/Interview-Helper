from __future__ import annotations

import inspect
import json
from collections.abc import Iterable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.contracts.events import (
    ClientEvent,
    ClientEventType,
    ServerEvent,
    ServerEventType,
)
from app.core.config import get_settings
from stt.base import StreamingTranscriber, TranscriptUpdate
from stt.factory import create_transcriber
from stt.whispercpp import WhisperCppError

router = APIRouter()


async def _send_event(
    websocket: WebSocket,
    *,
    event_type: ServerEventType,
    session_id: str,
    sequence: int | None = None,
    payload: dict[str, object] | None = None,
) -> None:
    await websocket.send_json(
        ServerEvent(
            type=event_type,
            session_id=session_id,
            sequence=sequence,
            payload=payload or {},
        ).model_dump(mode="json")
    )


async def _send_error(
    websocket: WebSocket,
    session_id: str,
    *,
    code: str,
    message: str,
    sequence: int | None = None,
) -> None:
    await _send_event(
        websocket,
        event_type=ServerEventType.ERROR,
        session_id=session_id,
        sequence=sequence,
        payload={"code": code, "message": message},
    )


async def _send_transcript_updates(
    websocket: WebSocket,
    session_id: str,
    updates: Iterable[TranscriptUpdate],
) -> None:
    for update in updates:
        await _send_event(
            websocket,
            event_type=(
                ServerEventType.TRANSCRIPT_FINAL
                if update.is_final
                else ServerEventType.TRANSCRIPT_PARTIAL
            ),
            session_id=session_id,
            payload={
                "text": update.text,
                "full_text": update.full_text,
                "is_final": update.is_final,
                "start_ms": update.start_ms,
                "end_ms": update.end_ms,
                "language": update.language,
            },
        )


@router.websocket("/ws/interview/{session_id}")
async def interview_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    await _send_event(
        websocket,
        event_type=ServerEventType.SESSION_READY,
        session_id=session_id,
        payload={
            "message": "Interview session is ready",
            "audio": {
                "encoding": "pcm_s16le",
                "sample_rate": 16_000,
                "channels": 1,
            },
        },
    )

    settings = get_settings()
    transcriber: StreamingTranscriber | None = None

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            audio = message.get("bytes")
            if audio is not None:
                if transcriber is None:
                    await _send_error(
                        websocket,
                        session_id,
                        code="stt_not_started",
                        message="Send start_session before audio frames",
                    )
                    continue
                try:
                    updates = await transcriber.push_audio(audio)
                except (ValueError, WhisperCppError) as exc:
                    await _send_error(
                        websocket,
                        session_id,
                        code="stt_failed",
                        message=str(exc),
                    )
                    continue
                await _send_transcript_updates(websocket, session_id, updates)
                continue

            raw_message = message.get("text")
            if raw_message is None:
                await _send_error(
                    websocket,
                    session_id,
                    code="unsupported_frame",
                    message="Only JSON text events and binary PCM16 audio are supported",
                )
                continue

            try:
                event = ClientEvent.model_validate_json(raw_message)
            except (ValidationError, json.JSONDecodeError) as exc:
                await _send_error(
                    websocket,
                    session_id,
                    code="invalid_event",
                    message=str(exc),
                )
                continue

            if event.session_id != session_id:
                await _send_error(
                    websocket,
                    session_id,
                    code="session_mismatch",
                    message="Event session_id does not match the WebSocket route",
                    sequence=event.sequence,
                )
                continue

            await _send_event(
                websocket,
                event_type=ServerEventType.EVENT_ACCEPTED,
                session_id=session_id,
                sequence=event.sequence,
                payload={"accepted_type": event.type},
            )

            if event.type == ClientEventType.START_SESSION:
                if transcriber is not None:
                    await transcriber.close()
                language_value = event.payload.get("language")
                language = str(language_value) if language_value else None
                try:
                    transcriber = create_transcriber(settings, language=language)
                    validator = getattr(transcriber, "validate_runtime", None)
                    if callable(validator):
                        validation_result = validator()
                        if inspect.isawaitable(validation_result):
                            await validation_result
                except (ValueError, WhisperCppError) as exc:
                    if transcriber is not None:
                        await transcriber.close()
                    transcriber = None
                    await _send_error(
                        websocket,
                        session_id,
                        code="stt_unavailable",
                        message=str(exc),
                        sequence=event.sequence,
                    )
                    continue
                await _send_event(
                    websocket,
                    event_type=ServerEventType.STT_READY,
                    session_id=session_id,
                    sequence=event.sequence,
                    payload={
                        "provider": settings.stt_provider,
                        "language": language or settings.whispercpp_language,
                        "encoding": "pcm_s16le",
                        "sample_rate": 16_000,
                        "channels": 1,
                    },
                )
                continue

            if event.type == ClientEventType.STOP_SESSION:
                if transcriber is not None:
                    try:
                        updates = await transcriber.flush()
                        await _send_transcript_updates(websocket, session_id, updates)
                    except (ValueError, WhisperCppError) as exc:
                        await _send_error(
                            websocket,
                            session_id,
                            code="stt_flush_failed",
                            message=str(exc),
                            sequence=event.sequence,
                        )
                    finally:
                        await transcriber.close()
                        transcriber = None
                await _send_event(
                    websocket,
                    event_type=ServerEventType.STT_STOPPED,
                    session_id=session_id,
                    sequence=event.sequence,
                    payload={"message": "Transcription stopped"},
                )
    except WebSocketDisconnect:
        pass
    finally:
        if transcriber is not None:
            await transcriber.close()
