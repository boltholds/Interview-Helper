from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.contracts.knowledge import KnowledgeSearchItem, KnowledgeSearchResponse
from app.core.config import get_settings
from ingestion.embeddings import EmbeddingProviderError, create_embedding_provider
from ingestion.index import SQLiteKnowledgeIndex
from ingestion.retrieval import HybridRetriever

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
SearchQuery = Annotated[str, Query(min_length=2, max_length=500)]
SearchLimit = Annotated[int, Query(ge=1, le=20)]


@router.get("/search", response_model=KnowledgeSearchResponse)
def search_knowledge(
    q: SearchQuery,
    limit: SearchLimit = 5,
    role: str | None = None,
    topic: str | None = None,
    level: str | None = None,
    language: str | None = None,
) -> KnowledgeSearchResponse:
    settings = get_settings()
    filters = {
        key: value
        for key, value in {
            "role": role,
            "topic": topic,
            "level": level,
            "language": language,
        }.items()
        if value
    }
    try:
        provider = create_embedding_provider(
            settings.embedding_provider,
            model=settings.embedding_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            local_dimensions=settings.local_embedding_dimensions,
            http_referer=settings.openrouter_http_referer,
            app_title=settings.openrouter_app_title,
        )
        results = HybridRetriever(
            SQLiteKnowledgeIndex(Path(settings.knowledge_index_path)),
            provider,
        ).search(q, limit=limit, filters=filters)
    except (EmbeddingProviderError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    items = [KnowledgeSearchItem(**result.to_dict()) for result in results]
    return KnowledgeSearchResponse(query=q, total=len(items), items=items)
