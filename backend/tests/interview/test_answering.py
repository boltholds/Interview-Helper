import asyncio

from app.core.config import Settings
from app.interview.answering import prepare_answer_context


def test_prepare_answer_context_falls_back_when_index_is_missing(tmp_path) -> None:
    settings = Settings(
        knowledge_index_path=str(tmp_path / "missing.db"),
        embedding_provider="local-hash",
        local_embedding_dimensions=64,
    )

    context = asyncio.run(
        prepare_answer_context(
            settings,
            question="Как устроен event loop?",
            role="Python Developer",
            language="ru",
        )
    )

    assert context.sources == []
    assert context.retrieval_warning
    assert context.messages[0].role == "system"
    assert "event loop" in context.messages[1].content
    assert "Python Developer" in context.messages[1].content
