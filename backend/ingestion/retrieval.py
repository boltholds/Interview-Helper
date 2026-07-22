from __future__ import annotations

import re
from dataclasses import dataclass, replace

from ingestion.embeddings import EmbeddingProvider
from ingestion.index import SQLiteKnowledgeIndex
from ingestion.models import SearchResult


@dataclass(frozen=True, slots=True)
class HybridWeights:
    semantic: float = 0.55
    bm25: float = 0.35
    metadata: float = 0.10

    def __post_init__(self) -> None:
        values = (self.semantic, self.bm25, self.metadata)
        if any(value < 0 for value in values):
            raise ValueError("Hybrid weights cannot be negative")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("Hybrid weights must sum to 1.0")


def _normalise_bm25(results: list[SearchResult]) -> dict[str, float]:
    positive = [result.bm25_score for result in results if result.bm25_score > 0]
    if not positive:
        return {}
    minimum = min(positive)
    maximum = max(positive)
    if maximum == minimum:
        return {
            result.chunk_id: 1.0
            for result in results
            if result.bm25_score > 0
        }
    return {
        result.chunk_id: (result.bm25_score - minimum) / (maximum - minimum)
        for result in results
        if result.bm25_score > 0
    }


def _semantic_to_unit_interval(score: float) -> float:
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _metadata_score(
    query: str,
    metadata: dict[str, str],
    filters: dict[str, str] | None,
) -> float:
    requested = {key: value for key, value in (filters or {}).items() if value}
    if requested:
        matched = sum(
            str(metadata.get(key, "")).casefold() == value.casefold()
            for key, value in requested.items()
        )
        return matched / len(requested)

    query_terms = set(re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE))
    metadata_terms = set(
        re.findall(
            r"[^\W_]+",
            " ".join(str(value) for value in metadata.values()).casefold(),
            flags=re.UNICODE,
        )
    )
    if not query_terms or not metadata_terms:
        return 0.0
    return min(1.0, len(query_terms & metadata_terms) / 2.0)


def _dedup_key(result: SearchResult) -> str:
    if result.content_hash:
        return result.content_hash
    return re.sub(r"\s+", " ", result.content.casefold()).strip()


class HybridRetriever:
    def __init__(
        self,
        index: SQLiteKnowledgeIndex,
        embedding_provider: EmbeddingProvider,
        *,
        weights: HybridWeights | None = None,
    ) -> None:
        self.index = index
        self.embedding_provider = embedding_provider
        self.weights = weights or HybridWeights()

    def _validate_provider(self) -> None:
        manifest = self.index.manifest()
        expected_provider = manifest.get("embedding_provider")
        expected_model = manifest.get("embedding_model")
        if not expected_provider or not expected_model:
            raise ValueError(
                "Knowledge index does not contain embedding metadata; rebuild the index"
            )
        if expected_provider != self.embedding_provider.name:
            raise ValueError(
                "Embedding provider mismatch: "
                f"index={expected_provider}, runtime={self.embedding_provider.name}"
            )
        if expected_model != self.embedding_provider.model:
            raise ValueError(
                "Embedding model mismatch: "
                f"index={expected_model}, runtime={self.embedding_provider.model}"
            )

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        if limit < 1:
            raise ValueError("limit must be positive")
        self._validate_provider()

        candidate_limit = max(limit * 8, 20)
        lexical = self.index.bm25_search(
            query,
            limit=candidate_limit,
            filters=filters,
        )
        query_vector = self.embedding_provider.embed_query(query)
        semantic = self.index.vector_search(
            query_vector,
            limit=candidate_limit,
            filters=filters,
        )

        merged: dict[str, SearchResult] = {}
        for result in lexical:
            merged[result.chunk_id] = result
        for result in semantic:
            existing = merged.get(result.chunk_id)
            if existing is None:
                merged[result.chunk_id] = result
            else:
                merged[result.chunk_id] = replace(
                    existing,
                    semantic_score=result.semantic_score,
                )

        bm25_by_id = _normalise_bm25(list(merged.values()))
        ranked: list[SearchResult] = []
        for result in merged.values():
            bm25_normalized = bm25_by_id.get(result.chunk_id, 0.0)
            semantic_normalized = _semantic_to_unit_interval(result.semantic_score)
            metadata_score = _metadata_score(query, result.metadata, filters)
            final_score = (
                self.weights.semantic * semantic_normalized
                + self.weights.bm25 * bm25_normalized
                + self.weights.metadata * metadata_score
            )
            ranked.append(
                replace(
                    result,
                    bm25_normalized=bm25_normalized,
                    semantic_normalized=semantic_normalized,
                    metadata_score=metadata_score,
                    final_score=final_score,
                )
            )

        ranked.sort(
            key=lambda item: (
                item.final_score,
                item.semantic_normalized,
                item.bm25_normalized,
            ),
            reverse=True,
        )

        deduplicated: list[SearchResult] = []
        seen: set[str] = set()
        for result in ranked:
            key = _dedup_key(result)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(result)
            if len(deduplicated) >= limit:
                break
        return deduplicated
