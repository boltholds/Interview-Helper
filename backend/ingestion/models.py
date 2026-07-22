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
    bm25_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_path": self.source_path,
            "title": self.title,
            "ordinal": self.ordinal,
            "metadata": self.metadata,
            "bm25_score": self.bm25_score,
        }
