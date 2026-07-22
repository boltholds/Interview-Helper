from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ingestion.models import SourceDocument

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".html", ".htm", ".json"}
_CONTENT_KEYS = ("content", "text", "body")
_BLOCK_TAGS = {
    "article",
    "br",
    "div",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "li",
    "main",
    "p",
    "pre",
    "section",
    "td",
    "th",
    "tr",
}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u200b", "")
    paragraphs = re.split(r"\n\s*\n", text)
    normalized = [re.sub(r"[ \t\f\v]+", " ", paragraph).strip() for paragraph in paragraphs]
    return "\n\n".join(paragraph for paragraph in normalized if paragraph)


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _relative_source(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def discover_source_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text

    closing = text.find("\n---\n", 4)
    if closing == -1:
        return {}, text

    metadata: dict[str, str] = {}
    for line in text[4:closing].splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip():
            metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, text[closing + 5 :]


def _markdown_to_text(text: str) -> str:
    text = re.sub(r"!\[([^]]*)]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(?:[-*+] |\d+[.)] )", "", text, flags=re.MULTILINE)
    text = text.replace("```", "").replace("~~~", "").replace("`", "")
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    return normalize_text(text)


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.metadata: dict[str, str] = {}
        self._ignored_depth = 0
        self._inside_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {key.lower(): value for key, value in attrs if value is not None}
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
            return
        if tag == "title":
            self._inside_title = True
        if tag == "meta":
            name = attributes.get("name") or attributes.get("property")
            content = attributes.get("content")
            if name and content:
                self.metadata[name.lower()] = content.strip()
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if tag == "title":
            self._inside_title = False
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._inside_title:
            self.title_parts.append(data)
        self.parts.append(data)


def _load_markdown(path: Path, root: Path) -> list[SourceDocument]:
    raw = path.read_text(encoding="utf-8")
    metadata, body = _parse_frontmatter(raw)
    content = _markdown_to_text(body)
    if not content:
        return []
    title = metadata.pop("title", path.stem.replace("-", " ").replace("_", " ").title())
    return [
        SourceDocument(
            content=content,
            source_path=_relative_source(path, root),
            title=title,
            metadata=metadata,
            content_hash=_hash_content(content),
        )
    ]


def _load_html(path: Path, root: Path) -> list[SourceDocument]:
    parser = _ReadableHTMLParser()
    parser.feed(path.read_text(encoding="utf-8"))
    content = normalize_text("".join(parser.parts))
    if not content:
        return []
    title = normalize_text(" ".join(parser.title_parts)) or path.stem
    allowed_metadata = {
        key: value
        for key, value in parser.metadata.items()
        if key in {"description", "keywords", "role", "topic", "level", "language"}
    }
    return [
        SourceDocument(
            content=content,
            source_path=_relative_source(path, root),
            title=title,
            metadata=allowed_metadata,
            content_hash=_hash_content(content),
        )
    ]


def _string_metadata(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
        return ", ".join(str(item) for item in value)
    return None


def _json_record_to_document(
    record: Any,
    *,
    path: Path,
    root: Path,
    position: int,
) -> SourceDocument | None:
    source_path = _relative_source(path, root)
    if position:
        source_path = f"{source_path}#item-{position}"

    if not isinstance(record, dict):
        content = normalize_text(str(record))
        if not content:
            return None
        return SourceDocument(
            content=content,
            source_path=source_path,
            title=f"{path.stem} item {position}",
            content_hash=_hash_content(content),
        )

    content_value = next((record[key] for key in _CONTENT_KEYS if record.get(key)), None)
    if content_value is None and record.get("question") and record.get("answer"):
        content_value = f"Question: {record['question']}\n\nAnswer: {record['answer']}"
    if content_value is None:
        return None

    content = normalize_text(str(content_value))
    if not content:
        return None

    title = str(record.get("title") or record.get("question") or path.stem).strip()
    metadata: dict[str, str] = {}
    nested_metadata = record.get("metadata")
    if isinstance(nested_metadata, dict):
        for key, value in nested_metadata.items():
            rendered = _string_metadata(value)
            if rendered is not None:
                metadata[str(key)] = rendered

    excluded = {*_CONTENT_KEYS, "question", "answer", "title", "metadata", "documents"}
    for key, value in record.items():
        if key in excluded:
            continue
        rendered = _string_metadata(value)
        if rendered is not None:
            metadata[str(key)] = rendered

    return SourceDocument(
        content=content,
        source_path=source_path,
        title=title,
        metadata=metadata,
        content_hash=_hash_content(content),
    )


def _load_json(path: Path, root: Path) -> list[SourceDocument]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("documents"), list):
        records = payload["documents"]
    elif isinstance(payload, list):
        records = payload
    else:
        records = [payload]

    documents = [
        _json_record_to_document(record, path=path, root=root, position=index)
        for index, record in enumerate(records)
    ]
    return [document for document in documents if document is not None]


def load_file(path: Path, root: Path) -> list[SourceDocument]:
    extension = path.suffix.lower()
    if extension in {".md", ".markdown"}:
        return _load_markdown(path, root)
    if extension in {".html", ".htm"}:
        return _load_html(path, root)
    if extension == ".json":
        return _load_json(path, root)
    return []


def load_documents(root: Path) -> tuple[list[SourceDocument], list[str]]:
    source_root = root if root.is_dir() else root.parent
    documents: list[SourceDocument] = []
    errors: list[str] = []
    for path in discover_source_files(root):
        try:
            documents.extend(load_file(path, source_root))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    return documents, errors
