from ingestion.chunker import ChunkingConfig, chunk_document
from ingestion.models import SourceDocument


def test_chunker_preserves_metadata_and_limits_size() -> None:
    document = SourceDocument(
        content="\n\n".join(f"Paragraph {index} " + "word " * 35 for index in range(8)),
        source_path="python/retries.md",
        title="Retries",
        metadata={"role": "Python Developer"},
    )

    chunks = chunk_document(
        document,
        ChunkingConfig(max_chars=300, overlap_chars=40, min_chars=40),
    )

    assert len(chunks) > 2
    assert all(chunk.metadata == {"role": "Python Developer"} for chunk in chunks)
    assert all(chunk.source_path == "python/retries.md" for chunk in chunks)
    assert all(len(chunk.content) <= 340 for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
