from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from ingestion.models import KnowledgeChunk, SearchResult


def _pack_vector(vector: Sequence[float]) -> bytes:
    if not vector:
        raise ValueError("Embedding vector cannot be empty")
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(blob: bytes, dimensions: int) -> list[float]:
    expected_size = dimensions * 4
    if len(blob) != expected_size:
        raise ValueError(
            f"Stored embedding has {len(blob)} bytes, expected {expected_size}"
        )
    return list(struct.unpack(f"<{dimensions}f", blob))


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"Embedding dimension mismatch: query={len(left)}, index={len(right)}"
        )
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _matches_filters(metadata: dict[str, str], filters: dict[str, str] | None) -> bool:
    requested = {
        key: value.casefold() for key, value in (filters or {}).items() if value
    }
    return not any(
        str(metadata.get(key, "")).casefold() != value
        for key, value in requested.items()
    )


class SQLiteKnowledgeIndex:
    def __init__(self, path: Path) -> None:
        self.path = path

    def build(
        self,
        chunks: Iterable[KnowledgeChunk],
        *,
        embeddings: dict[str, Sequence[float]] | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        manifest: dict[str, str | int] | None = None,
    ) -> int:
        chunk_list = list(chunks)
        chunk_ids = {chunk.chunk_id for chunk in chunk_list}
        if len(chunk_ids) != len(chunk_list):
            raise ValueError("Knowledge chunks contain duplicate chunk IDs")

        embedding_dimensions: int | None = None
        if embeddings is not None:
            if set(embeddings) != chunk_ids:
                missing = sorted(chunk_ids - set(embeddings))
                extra = sorted(set(embeddings) - chunk_ids)
                raise ValueError(
                    f"Embeddings must match chunks exactly; missing={missing}, extra={extra}"
                )
            dimensions = {len(vector) for vector in embeddings.values()}
            if len(dimensions) != 1 or 0 in dimensions:
                raise ValueError("All embedding vectors must have one non-zero dimension")
            embedding_dimensions = dimensions.pop()
            if not embedding_provider or not embedding_model:
                raise ValueError(
                    "embedding_provider and embedding_model are required with embeddings"
                )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)

        try:
            with sqlite3.connect(temporary_path) as connection:
                connection.executescript(
                    """
                    PRAGMA foreign_keys = ON;
                    CREATE TABLE chunks (
                        rowid INTEGER PRIMARY KEY,
                        chunk_id TEXT NOT NULL UNIQUE,
                        content TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        title TEXT NOT NULL,
                        ordinal INTEGER NOT NULL,
                        metadata_json TEXT NOT NULL,
                        content_hash TEXT NOT NULL
                    );
                    CREATE TABLE manifest (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE embeddings (
                        chunk_id TEXT PRIMARY KEY,
                        dimensions INTEGER NOT NULL,
                        vector BLOB NOT NULL,
                        FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
                    );
                    CREATE VIRTUAL TABLE chunks_fts USING fts5(
                        content,
                        title,
                        content='chunks',
                        content_rowid='rowid',
                        tokenize='unicode61 remove_diacritics 2'
                    );
                    """
                )
                connection.executemany(
                    """
                    INSERT INTO chunks (
                        chunk_id, content, source_path, title, ordinal, metadata_json, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.chunk_id,
                            chunk.content,
                            chunk.source_path,
                            chunk.title,
                            chunk.ordinal,
                            json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                            chunk.content_hash,
                        )
                        for chunk in chunk_list
                    ],
                )
                if embeddings is not None and embedding_dimensions is not None:
                    connection.executemany(
                        """
                        INSERT INTO embeddings (chunk_id, dimensions, vector)
                        VALUES (?, ?, ?)
                        """,
                        [
                            (
                                chunk.chunk_id,
                                embedding_dimensions,
                                _pack_vector(embeddings[chunk.chunk_id]),
                            )
                            for chunk in chunk_list
                        ],
                    )

                manifest_values: dict[str, str | int] = {
                    "schema_version": 2,
                    "chunk_count": len(chunk_list),
                    "embedding_count": len(embeddings or {}),
                }
                if embeddings is not None and embedding_dimensions is not None:
                    manifest_values.update(
                        {
                            "embedding_provider": embedding_provider or "",
                            "embedding_model": embedding_model or "",
                            "embedding_dimensions": embedding_dimensions,
                        }
                    )
                if manifest:
                    manifest_values.update(manifest)
                connection.executemany(
                    "INSERT INTO manifest (key, value) VALUES (?, ?)",
                    [(key, str(value)) for key, value in manifest_values.items()],
                )
                connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
                connection.commit()
            os.replace(temporary_path, self.path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return len(chunk_list)

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE)
        if not tokens:
            raise ValueError("Search query does not contain searchable terms")
        unique_tokens = list(dict.fromkeys(tokens))
        return " OR ".join(f'"{token}"' for token in unique_tokens)

    @staticmethod
    def _row_to_result(
        row: sqlite3.Row,
        *,
        bm25_score: float = 0.0,
        semantic_score: float = 0.0,
    ) -> SearchResult:
        return SearchResult(
            chunk_id=row["chunk_id"],
            content=row["content"],
            source_path=row["source_path"],
            title=row["title"],
            ordinal=row["ordinal"],
            metadata=json.loads(row["metadata_json"]),
            content_hash=row["content_hash"],
            bm25_score=bm25_score,
            semantic_score=semantic_score,
        )

    def bm25_search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if not self.path.exists():
            raise FileNotFoundError(f"Knowledge index does not exist: {self.path}")

        candidate_limit = max(limit * 8, limit)
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    chunks.chunk_id,
                    chunks.content,
                    chunks.source_path,
                    chunks.title,
                    chunks.ordinal,
                    chunks.metadata_json,
                    chunks.content_hash,
                    bm25(chunks_fts, 1.0, 0.25) AS rank
                FROM chunks_fts
                JOIN chunks ON chunks.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (self._fts_query(query), candidate_limit),
            ).fetchall()

        results: list[SearchResult] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if not _matches_filters(metadata, filters):
                continue
            results.append(
                self._row_to_result(
                    row,
                    bm25_score=max(0.0, -float(row["rank"])),
                )
            )
            if len(results) >= limit:
                break
        return results

    def vector_search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if not query_vector:
            raise ValueError("Query embedding cannot be empty")
        if not self.path.exists():
            raise FileNotFoundError(f"Knowledge index does not exist: {self.path}")

        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    chunks.chunk_id,
                    chunks.content,
                    chunks.source_path,
                    chunks.title,
                    chunks.ordinal,
                    chunks.metadata_json,
                    chunks.content_hash,
                    embeddings.dimensions,
                    embeddings.vector
                FROM embeddings
                JOIN chunks ON chunks.chunk_id = embeddings.chunk_id
                """
            ).fetchall()

        if not rows:
            raise ValueError(
                "Knowledge index does not contain embeddings; rebuild it with an embedding provider"
            )

        ranked: list[SearchResult] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if not _matches_filters(metadata, filters):
                continue
            vector = _unpack_vector(row["vector"], int(row["dimensions"]))
            ranked.append(
                self._row_to_result(
                    row,
                    semantic_score=_cosine_similarity(query_vector, vector),
                )
            )
        ranked.sort(key=lambda item: item.semantic_score, reverse=True)
        return ranked[:limit]

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """Backward-compatible alias for the lexical search layer."""
        return self.bm25_search(query, limit=limit, filters=filters)

    def manifest(self) -> dict[str, str]:
        if not self.path.exists():
            raise FileNotFoundError(f"Knowledge index does not exist: {self.path}")
        with sqlite3.connect(self.path) as connection:
            return dict(connection.execute("SELECT key, value FROM manifest").fetchall())
