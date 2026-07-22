from pathlib import Path

from ingestion.index import SQLiteKnowledgeIndex
from ingestion.service import build_knowledge_index


def test_build_service_creates_searchable_index(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "jobs.md").write_text(
        "---\nrole: Python Developer\ntopic: queues\n---\n\n"
        "An idempotent consumer can safely process the same event more than once.",
        encoding="utf-8",
    )
    output = tmp_path / "data" / "knowledge.db"

    report = build_knowledge_index(sources, output)
    results = SQLiteKnowledgeIndex(output).search("idempotent event")

    assert report.documents == 1
    assert report.chunks == 1
    assert report.errors == []
    assert results[0].metadata["topic"] == "queues"
