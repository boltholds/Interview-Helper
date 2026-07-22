import json
from pathlib import Path

from ingestion.loaders import load_documents


def test_loads_markdown_html_and_json(tmp_path: Path) -> None:
    (tmp_path / "python.md").write_text(
        "---\ntitle: Python retries\nrole: Python Developer\n---\n\n"
        "# Retry\n\nUse backoff.",
        encoding="utf-8",
    )
    (tmp_path / "database.html").write_text(
        "<html><head><title>Indexes</title>"
        "<meta name='topic' content='databases'></head>"
        "<body><script>ignore me</script><h1>B-tree</h1>"
        "<p>Indexes speed up reads.</p></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "questions.json").write_text(
        json.dumps(
            [
                {
                    "question": "What is idempotency?",
                    "answer": "Repeated execution has the same effect.",
                    "metadata": {"level": "middle"},
                },
                {"content": "Use transactions for atomic updates.", "topic": "databases"},
            ]
        ),
        encoding="utf-8",
    )

    documents, errors = load_documents(tmp_path)

    assert errors == []
    assert len(documents) == 4
    assert documents[0].title == "Indexes"
    assert documents[0].metadata["topic"] == "databases"
    assert all("ignore me" not in document.content for document in documents)
    assert any(document.metadata.get("role") == "Python Developer" for document in documents)
    assert any("Question: What is idempotency?" in document.content for document in documents)
