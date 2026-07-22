from pathlib import Path

from ingestion.index import SQLiteKnowledgeIndex
from ingestion.models import KnowledgeChunk


def _chunk(chunk_id: str, content: str, *, role: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        content=content,
        source_path=f"{chunk_id}.md",
        title=chunk_id,
        ordinal=0,
        metadata={"role": role, "language": "en"},
        content_hash=f"hash-{chunk_id}",
    )


def test_builds_and_searches_fts_index(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.db"
    index = SQLiteKnowledgeIndex(path)
    index.build(
        [
            _chunk(
                "python",
                "Idempotent workers use retries and exponential backoff.",
                role="Python Developer",
            ),
            _chunk(
                "frontend",
                "React components render a browser interface.",
                role="Frontend Developer",
            ),
        ]
    )

    results = index.search("worker retries", filters={"role": "Python Developer"})

    assert path.exists()
    assert [result.chunk_id for result in results] == ["python"]
    assert float(results[0].bm25_score) > 0
    assert index.manifest()["chunk_count"] == "2"
