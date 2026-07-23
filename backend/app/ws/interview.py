from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Iterable
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.contracts.events import (
    ClientEvent,
    ClientEventType,
    ServerEvent,
    ServerEventType,
)
from app.core.config import Settings, get_settings
from app.interview.answering import prepare_answer_context
from app.interview.question_detector import QuestionDetector
from llm.factory import create_llm_provider
from llm.openrouter import OpenRouterError
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
    send_lock: asyncio.Lock | None = None,
) -> None:
    data = ServerEvent(
        type=event_type,
        session_id=session_id,
        sequence=sequence,
        payload=payload or {},
    ).model_dump(mode="json")
    if send_lock is None:
        await websocket.send_json(data)
        return
    async with send_lock:
        await websocket.send_json(data)


async def _send_error(
    websocket: WebSocket,
    session_id: str,
    *,
    code: str,
    message: str,
    sequence: int | None = None,
    send_lock: asyncio.Lock | None = None,
) -> None:
    await _send_event(
        websocket,
        event_type=ServerEventType.ERROR,
        session_id=session_id,
        sequence=sequence,
        payload={"code": code, "message": message},
        send_lock=send_lock,
    )


async def _send_transcript_updates(
    websocket: WebSocket,
    session_id: str,
    updates: Iterable[TranscriptUpdate],
    *,
    detector: QuestionDetector,
    send_lock: asyncio.Lock,
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
            send_lock=send_lock,
        )
        if not update.is_final:
            continue
        question = detector.feed(update.full_text)
        if question is None:
            continue
        await _send_event(
            websocket,
            event_type=ServerEventType.QUESTION_DETECTED,
            session_id=session_id,
            payload={
                "question": question.text,
                "confidence": question.confidence,
                "reason": question.reason,
                "automatic": True,
            },
            send_lock=send_lock,
        )


async def _run_generation(
    websocket: WebSocket,
    *,
    session_id: str,
    settings: Settings,
    question: str,
    role: str,
    language: str,
    send_lock: asyncio.Lock,
) -> None:
    provider = None
    try:
        context = await prepare_answer_context(
            settings,
            question=question,
            role=role,
            language=language,
        )
        await _send_event(
            websocket,
            event_type=ServerEventType.GENERATION_STARTED,
            session_id=session_id,
            payload={
                "question": question,
                "sources": context.sources,
                "retrieval_warning": context.retrieval_warning,
            },
            send_lock=send_lock,
        )

        provider = create_llm_provider(settings)
        answer = ""
        async for delta in provider.stream(context.messages):
            answer += delta
            await _send_event(
                websocket,
                event_type=ServerEventType.ANSWER_DELTA,
                session_id=session_id,
                payload={"delta": delta, "text": answer},
                send_lock=send_lock,
            )
        await _send_event(
            websocket,
            event_type=ServerEventType.ANSWER_COMPLETED,
            session_id=session_id,
            payload={
                "question": question,
                "answer": answer,
                "sources": context.sources,
            },
            send_lock=send_lock,
        )
    except asyncio.CancelledError:
        raise
    except (OpenRouterError, ValueError) as exc:
        await _send_error(
            websocket,
            session_id,
            code="generation_failed",
            message=str(exc),
            send_lock=send_lock,
        )
    finally:
        if provider is not None:
            await provider.close()


async def _cancel_generation(task: asyncio.Task[None] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@router.websocket("/ws/interview/{session_id}")
async def interview_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    send_lock = asyncio.Lock()
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
        send_lock=send_lock,
    )

    settings = get_settings()
    transcriber: StreamingTranscriber | None = None
    detector = QuestionDetector()
    generation_task: asyncio.Task[None] | None = None
    role = "Python Developer"
    language = settings.whispercpp_language

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
                        send_lock=send_lock,
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
                        send_lock=send_lock,
                    )
                    continue
                await _send_transcript_updates(
                    websocket,
                    session_id,
                    updates,
                    detector=detector,
                    send_lock=send_lock,
                )
                continue

            raw_message = message.get("text")
            if raw_message is None:
                await _send_error(
                    websocket,
                    session_id,
                    code="unsupported_frame",
                    message="Only JSON text events and binary PCM16 audio are supported",
                    send_lock=send_lock,
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
                    send_lock=send_lock,
                )
                continue

            if event.session_id != session_id:
                await _send_error(
                    websocket,
                    session_id,
                    code="session_mismatch",
                    message="Event session_id does not match the WebSocket route",
                    sequence=event.sequence,
                    send_lock=send_lock,
                )
                continue

            await _send_event(
                websocket,
                event_type=ServerEventType.EVENT_ACCEPTED,
                session_id=session_id,
                sequence=event.sequence,
                payload={"accepted_type": event.type},
                send_lock=send_lock,
            )

            if event.type == ClientEventType.START_SESSION:
                if transcriber is not None:
                    await transcriber.close()
                detector.reset()
                await _cancel_generation(generation_task)
                generation_task = None
                language_value = event.payload.get("language")
                language = str(language_value) if language_value else settings.whispercpp_language
                role_value = event.payload.get("role")
                role = str(role_value).strip() if role_value else role
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
                        send_lock=send_lock,
                    )
                    continue
                await _send_event(
                    websocket,
                    event_type=ServerEventType.STT_READY,
                    session_id=session_id,
                    sequence=event.sequence,
                    payload={
                        "provider": settings.stt_provider,
                        "language": language,
                        "encoding": "pcm_s16le",
                        "sample_rate": 16_000,
                        "channels": 1,
                    },
                    send_lock=send_lock,
                )
                continue

            if event.type == ClientEventType.COMMIT_QUESTION:
                raw_question = event.payload.get("question")
                question = " ".join(str(raw_question or "").split())
                if len(question) < 2:
                    await _send_error(
                        websocket,
                        session_id,
                        code="invalid_question",
                        message="Question must contain at least two characters",
                        sequence=event.sequence,
                        send_lock=send_lock,
                    )
                    continue
                await _cancel_generation(generation_task)
                await _send_event(
                    websocket,
                    event_type=ServerEventType.QUESTION_DETECTED,
                    session_id=session_id,
                    sequence=event.sequence,
                    payload={
                        "question": question,
                        "confidence": 1.0,
                        "reason": "manual_commit",
                        "automatic": False,
                    },
                    send_lock=send_lock,
                )
                generation_task = asyncio.create_task(
                    _run_generation(
                        websocket,
                        session_id=session_id,
                        settings=settings,
                        question=question,
                        role=role,
                        language=language,
                        send_lock=send_lock,
                    )
                )
                continue

            if event.type == ClientEventType.CANCEL_GENERATION:
                await _cancel_generation(generation_task)
                generation_task = None
                continue

            if event.type == ClientEventType.STOP_SESSION:
                if transcriber is not None:
                    try:
                        updates = await transcriber.flush()
                        await _send_transcript_updates(
                            websocket,
                            session_id,
                            updates,
                            detector=detector,
                            send_lock=send_lock,
                        )
                    except (ValueError, WhisperCppError) as exc:
                        await _send_error(
                            websocket,
                            session_id,
                            code="stt_flush_failed",
                            message=str(exc),
                            sequence=event.sequence,
                            send_lock=send_lock,
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
                    send_lock=send_lock,
                )
    except WebSocketDisconnect:
        pass
    finally:
        await _cancel_generation(generation_task)
        if transcriber is not None:
            await transcriber.close()
