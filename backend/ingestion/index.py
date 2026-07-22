from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterable

from ingestion.models import KnowledgeChunk, SearchResult


class SQLiteKnowledgeIndex:
    def __init__(self, path: Path) -> None:
        self.path = path

    def build(
        self,
        chunks: Iterable[KnowledgeChunk],
        *,
        manifest: dict[str, str | int] | None = None,
    ) -> int:
        chunk_list = list(chunks)
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
                manifest_values = {"schema_version": 1, "chunk_count": len(chunk_list)}
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

    def search(
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
                    bm25(chunks_fts, 1.0, 0.25) AS rank
                FROM chunks_fts
                JOIN chunks ON chunks.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (self._fts_query(query), candidate_limit),
            ).fetchall()

        requested_filters = {key: value.casefold() for key, value in (filters or {}).items() if value}
        results: list[SearchResult] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if any(str(metadata.get(key, "")).casefold() != value for key, value in requested_filters.items()):
                continue
            results.append(
                SearchResult(
                    chunk_id=row["chunk_id"],
                    content=row["content"],
                    source_path=row["source_path"],
                    title=row["title"],
                    ordinal=row["ordinal"],
                    metadata=metadata,
                    bm25_score=-float(row["rank"]),
                )
            )
            if len(results) >= limit:
                break
        return results

    def manifest(self) -> dict[str, str]:
        if not self.path.exists():
            raise FileNotFoundError(f"Knowledge index does not exist: {self.path}")
        with sqlite3.connect(self.path) as connection:
            return dict(connection.execute("SELECT key, value FROM manifest").fetchall())
