from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ingestion.models import KnowledgeChunk, SourceDocument


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    max_chars: int = 1200
    overlap_chars: int = 180
    min_chars: int = 120

    def __post_init__(self) -> None:
        if self.max_chars < 200:
            raise ValueError("max_chars must be at least 200")
        if self.overlap_chars < 0 or self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be between 0 and max_chars")
        if self.min_chars < 1 or self.min_chars > self.max_chars:
            raise ValueError("min_chars must be between 1 and max_chars")


def _tail(text: str, size: int) -> str:
    if size <= 0 or len(text) <= size:
        return text if size > 0 else ""
    candidate = text[-size:]
    boundary = candidate.find(" ")
    return candidate[boundary + 1 :].strip() if boundary >= 0 else candidate.strip()


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    segments: list[str] = []
    current: list[str] = []
    current_size = 0
    for word in words:
        projected = current_size + len(word) + (1 if current else 0)
        if current and projected > max_chars:
            segment = " ".join(current)
            segments.append(segment)
            overlap = _tail(segment, overlap_chars).split()
            current = overlap
            current_size = len(" ".join(current))
        current.append(word)
        current_size += len(word) + (1 if len(current) > 1 else 0)
    if current:
        segments.append(" ".join(current))
    return segments


def _segments(content: str, config: ChunkingConfig) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", content) if paragraph.strip()]
    output: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= config.max_chars:
            output.append(paragraph)
        else:
            output.extend(_split_long_text(paragraph, config.max_chars, config.overlap_chars))
    return output


def _pack_segments(segments: list[str], config: ChunkingConfig) -> list[str]:
    chunks: list[str] = []
    current = ""
    for segment in segments:
        candidate = f"{current}\n\n{segment}" if current else segment
        if len(candidate) <= config.max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        overlap = _tail(current, config.overlap_chars)
        candidate = f"{overlap}\n\n{segment}" if overlap else segment
        current = candidate if len(candidate) <= config.max_chars else segment

    if current:
        chunks.append(current)

    if len(chunks) > 1 and len(chunks[-1]) < config.min_chars:
        merged = f"{chunks[-2]}\n\n{chunks[-1]}"
        if len(merged) <= config.max_chars + config.overlap_chars:
            chunks[-2:] = [merged]
    return chunks


def chunk_document(
    document: SourceDocument,
    config: ChunkingConfig | None = None,
) -> list[KnowledgeChunk]:
    active_config = config or ChunkingConfig()
    texts = _pack_segments(_segments(document.content, active_config), active_config)
    chunks: list[KnowledgeChunk] = []
    for ordinal, content in enumerate(texts):
        fingerprint = hashlib.sha256(
            f"{document.source_path}:{ordinal}:{content}".encode("utf-8")
        ).hexdigest()
        chunks.append(
            KnowledgeChunk(
                chunk_id=fingerprint[:24],
                content=content,
                source_path=document.source_path,
                title=document.title,
                ordinal=ordinal,
                metadata=dict(document.metadata),
                content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        )
    return chunks
