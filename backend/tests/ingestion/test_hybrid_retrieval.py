from collections.abc import Sequence
from pathlib import Path

from ingestion.embeddings import EmbeddingProvider
from ingestion.index import SQLiteKnowledgeIndex
from ingestion.models import KnowledgeChunk
from ingestion.retrieval import HybridRetriever


class StaticEmbeddingProvider(EmbeddingProvider):
    name = "static"
    model = "static-v1"

    @property
    def dimensions(self) -> int:
        return 2

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        mapping = {
            "reliable jobs": [1.0, 0.0],
            "Idempotent workers use retries.": [1.0, 0.0],
            "Duplicate wording": [0.9, 0.1],
            "React components render UI.": [0.0, 1.0],
        }
        return [mapping[text] for text in texts]


def _chunk(
    chunk_id: str,
    content: str,
    *,
    role: str,
    content_hash: str | None = None,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        content=content,
        source_path=f"{chunk_id}.md",
        title=chunk_id,
        ordinal=0,
        metadata={"role": role, "language": "en"},
        content_hash=content_hash or f"hash-{chunk_id}",
    )


def test_hybrid_retrieval_combines_scores_filters_and_deduplicates(tmp_path: Path) -> None:
    provider = StaticEmbeddingProvider()
    chunks = [
        _chunk(
            "python",
            "Idempotent workers use retries.",
            role="Python Developer",
            content_hash="same-content",
        ),
        _chunk(
            "duplicate",
            "Duplicate wording",
            role="Python Developer",
            content_hash="same-content",
        ),
        _chunk(
            "frontend",
            "React components render UI.",
            role="Frontend Developer",
        ),
    ]
    vectors = provider.embed_documents([chunk.content for chunk in chunks])
    index = SQLiteKnowledgeIndex(tmp_path / "knowledge.db")
    index.build(
        chunks,
        embeddings={
            chunk.chunk_id: vector
            for chunk, vector in zip(chunks, vectors, strict=True)
        },
        embedding_provider=provider.name,
        embedding_model=provider.model,
    )

    results = HybridRetriever(index, provider).search(
        "reliable jobs",
        limit=5,
        filters={"role": "Python Developer"},
    )

    assert len(results) == 1
    assert results[0].chunk_id == "python"
    assert results[0].semantic_score > 0.9
    assert 0 <= results[0].final_score <= 1


def test_hybrid_retrieval_rejects_provider_mismatch(tmp_path: Path) -> None:
    provider = StaticEmbeddingProvider()
    chunk = _chunk("python", "Idempotent workers use retries.", role="Python Developer")
    index = SQLiteKnowledgeIndex(tmp_path / "knowledge.db")
    index.build(
        [chunk],
        embeddings={chunk.chunk_id: [1.0, 0.0]},
        embedding_provider="another-provider",
        embedding_model=provider.model,
    )

    try:
        HybridRetriever(index, provider).search("reliable jobs")
    except ValueError as exc:
        assert "provider mismatch" in str(exc).casefold()
    else:
        raise AssertionError("Expected provider mismatch")
