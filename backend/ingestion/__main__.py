from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from ingestion.chunker import ChunkingConfig
from ingestion.embeddings import create_embedding_provider
from ingestion.index import SQLiteKnowledgeIndex
from ingestion.retrieval import HybridRetriever
from ingestion.service import build_knowledge_index


def _add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-provider",
        default=os.getenv("EMBEDDING_PROVIDER", "openrouter"),
        choices=["local-hash", "openrouter", "openai"],
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b"),
    )
    parser.add_argument(
        "--local-embedding-dimensions",
        type=int,
        default=int(os.getenv("LOCAL_EMBEDDING_DIMENSIONS", "256")),
    )
    parser.add_argument(
        "--embedding-base-url",
        default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


def _provider(arguments: argparse.Namespace):
    selected = arguments.embedding_provider
    api_key = (
        os.getenv("OPENROUTER_API_KEY")
        if selected == "openrouter"
        else os.getenv("OPENAI_API_KEY")
    )
    return create_embedding_provider(
        selected,
        model=arguments.embedding_model,
        api_key=api_key,
        base_url=arguments.embedding_base_url,
        local_dimensions=arguments.local_embedding_dimensions,
        http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
        app_title=os.getenv("OPENROUTER_APP_TITLE", "Interview Helper"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ingestion")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-index", help="Build a local hybrid search index")
    build.add_argument("source_dir", type=Path)
    build.add_argument("--output", type=Path, default=Path("data/index/knowledge.db"))
    build.add_argument("--max-chars", type=int, default=1200)
    build.add_argument("--overlap-chars", type=int, default=180)
    build.add_argument("--min-chars", type=int, default=120)
    _add_embedding_arguments(build)

    search = commands.add_parser("search", help="Search an existing hybrid index")
    search.add_argument("query")
    search.add_argument("--index", type=Path, default=Path("data/index/knowledge.db"))
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--role")
    search.add_argument("--topic")
    search.add_argument("--level")
    search.add_argument("--language")
    _add_embedding_arguments(search)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    provider = _provider(arguments)
    if arguments.command == "build-index":
        report = build_knowledge_index(
            arguments.source_dir,
            arguments.output,
            chunking=ChunkingConfig(
                max_chars=arguments.max_chars,
                overlap_chars=arguments.overlap_chars,
                min_chars=arguments.min_chars,
            ),
            embedding_provider=provider,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0

    filters = {
        key: value
        for key, value in {
            "role": arguments.role,
            "topic": arguments.topic,
            "level": arguments.level,
            "language": arguments.language,
        }.items()
        if value
    }
    results = HybridRetriever(
        SQLiteKnowledgeIndex(arguments.index),
        provider,
    ).search(
        arguments.query,
        limit=arguments.limit,
        filters=filters,
    )
    print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
