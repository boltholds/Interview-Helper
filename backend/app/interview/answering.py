from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from ingestion.embeddings import EmbeddingProviderError, create_embedding_provider
from ingestion.index import SQLiteKnowledgeIndex
from ingestion.models import SearchResult
from ingestion.retrieval import HybridRetriever
from llm.base import ChatMessage


@dataclass(frozen=True, slots=True)
class AnswerContext:
    messages: list[ChatMessage]
    sources: list[dict[str, object]]
    retrieval_warning: str | None = None


def _search_knowledge(
    settings: Settings,
    question: str,
    role: str,
    language: str,
) -> list[SearchResult]:
    provider = create_embedding_provider(
        settings.embedding_provider,
        model=settings.embedding_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        local_dimensions=settings.local_embedding_dimensions,
        http_referer=settings.openrouter_http_referer,
        app_title=settings.openrouter_app_title,
    )
    retriever = HybridRetriever(
        SQLiteKnowledgeIndex(Path(settings.knowledge_index_path)),
        provider,
    )
    filters = {
        key: value
        for key, value in {
            "role": role,
            "language": language if language and language != "auto" else None,
        }.items()
        if value
    }
    results = retriever.search(question, limit=5, filters=filters)
    if not results and filters:
        results = retriever.search(question, limit=5)
    return results


def _source_payload(result: SearchResult) -> dict[str, object]:
    return {
        "title": result.title,
        "source_path": result.source_path,
        "score": round(result.final_score, 4),
        "metadata": result.metadata,
    }


def _context_text(results: list[SearchResult]) -> str:
    if not results:
        return "Контекст из базы знаний не найден. Используй общие профессиональные знания."
    snippets = []
    for index, result in enumerate(results, start=1):
        content = result.content.strip()
        if len(content) > 1_800:
            content = f"{content[:1_800].rstrip()}…"
        snippets.append(f"[{index}] {result.title}\n{content}")
    return "\n\n".join(snippets)


def _build_messages(
    *,
    question: str,
    role: str,
    language: str,
    results: list[SearchResult],
) -> list[ChatMessage]:
    response_language = (
        "языке вопроса"
        if not language or language == "auto"
        else f"языке с кодом {language}"
    )
    system = (
        "Ты ИИ-помощник кандидата на техническом собеседовании. "
        "Дай точную, практичную и честную подсказку без выдуманных фактов. "
        "Сначала сформулируй 3–6 коротких тезисов, затем предложи связный ответ, "
        "который можно произнести вслух. Если контекста недостаточно, явно обозначь допущения. "
        f"Отвечай на {response_language}."
    )
    user = (
        f"Целевая роль: {role or 'не указана'}\n"
        f"Вопрос интервьюера: {question}\n\n"
        "Релевантный контекст:\n"
        f"{_context_text(results)}\n\n"
        "Сформируй подсказку для ответа кандидата. Не упоминай, что используешь базу знаний."
    )
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]


async def prepare_answer_context(
    settings: Settings,
    *,
    question: str,
    role: str,
    language: str,
) -> AnswerContext:
    warning: str | None = None
    try:
        results = await asyncio.to_thread(
            _search_knowledge,
            settings,
            question,
            role,
            language,
        )
    except (EmbeddingProviderError, FileNotFoundError, OSError, ValueError) as exc:
        results = []
        warning = str(exc)

    return AnswerContext(
        messages=_build_messages(
            question=question,
            role=role,
            language=language,
            results=results,
        ),
        sources=[_source_payload(result) for result in results],
        retrieval_warning=warning,
    )
