from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ingestion.chunker import ChunkingConfig
from ingestion.index import SQLiteKnowledgeIndex
from ingestion.service import build_knowledge_index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ingestion")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-index", help="Build a local SQLite FTS5 index")
    build.add_argument("source_dir", type=Path)
    build.add_argument("--output", type=Path, default=Path("data/index/knowledge.db"))
    build.add_argument("--max-chars", type=int, default=1200)
    build.add_argument("--overlap-chars", type=int, default=180)
    build.add_argument("--min-chars", type=int, default=120)

    search = commands.add_parser("search", help="Search an existing local index")
    search.add_argument("query")
    search.add_argument("--index", type=Path, default=Path("data/index/knowledge.db"))
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--role")
    search.add_argument("--topic")
    search.add_argument("--language")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build-index":
        report = build_knowledge_index(
            arguments.source_dir,
            arguments.output,
            chunking=ChunkingConfig(
                max_chars=arguments.max_chars,
                overlap_chars=arguments.overlap_chars,
                min_chars=arguments.min_chars,
            ),
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0

    filters = {
        key: value
        for key, value in {
            "role": arguments.role,
            "topic": arguments.topic,
            "language": arguments.language,
        }.items()
        if value
    }
    results = SQLiteKnowledgeIndex(arguments.index).search(
        arguments.query,
        limit=arguments.limit,
        filters=filters,
    )
    print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
