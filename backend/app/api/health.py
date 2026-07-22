from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["system"])


@router.get("/health")
def healthcheck() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
        "stt_provider": settings.stt_provider,
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
    }
