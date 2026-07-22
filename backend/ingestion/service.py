from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ingestion.chunker import ChunkingConfig, chunk_document
from ingestion.index import SQLiteKnowledgeIndex
from ingestion.loaders import discover_source_files, load_documents


@dataclass(frozen=True, slots=True)
class BuildReport:
    source_dir: str
    output_path: str
    source_files: int
    documents: int
    chunks: int
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_knowledge_index(
    source_dir: Path,
    output_path: Path,
    *,
    chunking: ChunkingConfig | None = None,
) -> BuildReport:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_dir}")

    documents, errors = load_documents(source_dir)
    if not documents:
        detail = "; ".join(errors) if errors else "no supported non-empty documents found"
        raise ValueError(f"Cannot build knowledge index: {detail}")

    active_chunking = chunking or ChunkingConfig()
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, active_chunking)
    ]
    if not chunks:
        raise ValueError("Cannot build knowledge index: chunking produced no chunks")

    SQLiteKnowledgeIndex(output_path).build(
        chunks,
        manifest={
            "built_at": datetime.now(UTC).isoformat(),
            "source_path": str(source_dir),
            "document_count": len(documents),
        },
    )
    return BuildReport(
        source_dir=str(source_dir),
        output_path=str(output_path),
        source_files=len(discover_source_files(source_dir)),
        documents=len(documents),
        chunks=len(chunks),
        errors=errors,
    )
