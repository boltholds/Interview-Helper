import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app.api.health import router as health_router
from app.contracts.events import ClientEvent, ServerEvent, ServerEventType
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Interview Helper MVP backend",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router, prefix=settings.api_prefix)


@app.websocket("/ws/interview/{session_id}")
async def interview_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    await websocket.send_json(
        ServerEvent(
            type=ServerEventType.SESSION_READY,
            session_id=session_id,
            payload={"message": "Interview session is ready"},
        ).model_dump(mode="json")
    )

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                event = ClientEvent.model_validate_json(raw_message)
            except (ValidationError, json.JSONDecodeError) as exc:
                await websocket.send_json(
                    ServerEvent(
                        type=ServerEventType.ERROR,
                        session_id=session_id,
                        payload={"code": "invalid_event", "message": str(exc)},
                    ).model_dump(mode="json")
                )
                continue

            await websocket.send_json(
                ServerEvent(
                    type=ServerEventType.EVENT_ACCEPTED,
                    session_id=session_id,
                    sequence=event.sequence,
                    payload={"accepted_type": event.type},
                ).model_dump(mode="json")
            )
    except WebSocketDisconnect:
        return
