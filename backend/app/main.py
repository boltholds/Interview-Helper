from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.knowledge import router as knowledge_router
from app.core.config import get_settings
from app.ws.interview import router as interview_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.4.0",
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
app.include_router(knowledge_router, prefix=settings.api_prefix)
app.include_router(interview_router)
