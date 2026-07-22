from pathlib import Path

from ingestion.index import SQLiteKnowledgeIndex
from ingestion.models import KnowledgeChunk


def _chunk(chunk_id: str, content: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        content=content,
        source_path=f"{chunk_id}.md",
        title=chunk_id,
        ordinal=0,
        metadata={"role": "Python Developer"},
        content_hash=f"hash-{chunk_id}",
    )


def test_vector_search_uses_cosine_similarity(tmp_path: Path) -> None:
    index = SQLiteKnowledgeIndex(tmp_path / "knowledge.db")
    chunks = [_chunk("near", "near"), _chunk("far", "far")]
    index.build(
        chunks,
        embeddings={"near": [1.0, 0.0], "far": [0.0, 1.0]},
        embedding_provider="static",
        embedding_model="static-v1",
    )

    results = index.vector_search([0.9, 0.1], limit=2)

    assert [result.chunk_id for result in results] == ["near", "far"]
    assert index.manifest()["embedding_count"] == "2"
    assert index.manifest()["embedding_dimensions"] == "2"
