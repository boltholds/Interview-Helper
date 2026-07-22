from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status

from app.contracts.knowledge import KnowledgeSearchItem, KnowledgeSearchResponse
from app.core.config import get_settings
from ingestion.index import SQLiteKnowledgeIndex

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/search", response_model=KnowledgeSearchResponse)
def search_knowledge(
    q: str = Query(min_length=2, max_length=500),
    limit: int = Query(default=5, ge=1, le=20),
    role: str | None = None,
    topic: str | None = None,
    language: str | None = None,
) -> KnowledgeSearchResponse:
    settings = get_settings()
    index = SQLiteKnowledgeIndex(Path(settings.knowledge_index_path))
    filters = {
        key: value
        for key, value in {"role": role, "topic": topic, "language": language}.items()
        if value
    }
    try:
        results = index.search(q, limit=limit, filters=filters)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    items = [KnowledgeSearchItem(**result.to_dict()) for result in results]
    return KnowledgeSearchResponse(query=q, total=len(items), items=items)
