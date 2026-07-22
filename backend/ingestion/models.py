from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceDocument:
    content: str
    source_path: str
    title: str
    metadata: dict[str, str] = field(default_factory=dict)
    content_hash: str = ""


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    chunk_id: str
    content: str
    source_path: str
    title: str
    ordinal: int
    metadata: dict[str, str]
    content_hash: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk_id: str
    content: str
    source_path: str
    title: str
    ordinal: int
    metadata: dict[str, str]
    content_hash: str = ""
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    bm25_normalized: float = 0.0
    semantic_normalized: float = 0.0
    metadata_score: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_path": self.source_path,
            "title": self.title,
            "ordinal": self.ordinal,
            "metadata": self.metadata,
            "bm25_score": self.bm25_score,
            "semantic_score": self.semantic_score,
            "bm25_normalized": self.bm25_normalized,
            "semantic_normalized": self.semantic_normalized,
            "metadata_score": self.metadata_score,
            "final_score": self.final_score,
        }
