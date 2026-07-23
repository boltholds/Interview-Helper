from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DetectedQuestion:
    text: str
    confidence: float
    reason: str


_EXPLICIT_QUESTION_MIN_CHARS = 8
_MAX_QUESTION_CHARS = 500
_QUESTION_CUES = (
    "кто",
    "что",
    "где",
    "когда",
    "почему",
    "зачем",
    "как",
    "какой",
    "какая",
    "какие",
    "какое",
    "сколько",
    "можете",
    "могли бы",
    "расскажите",
    "опишите",
    "объясните",
    "приведите",
    "представьте",
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "which",
    "can you",
    "could you",
    "would you",
    "do you",
    "does",
    "did",
    "is",
    "are",
    "tell me",
    "describe",
    "explain",
    "give an example",
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _dedup_key(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE).strip()


def _trim_to_sentence_start(text: str) -> str:
    starts = [text.rfind(marker) for marker in (". ", "! ", "? ", "\n")]
    start = max(starts)
    return text[start + 1 :].strip() if start >= 0 else text.strip()


def _last_explicit_question(text: str) -> str | None:
    question_mark = text.rfind("?")
    if question_mark < 0:
        return None
    candidate = _trim_to_sentence_start(text[: question_mark + 1])
    if not (_EXPLICIT_QUESTION_MIN_CHARS <= len(candidate) <= _MAX_QUESTION_CHARS):
        return None
    return candidate


def _last_sentence(text: str) -> str:
    stripped = text.rstrip(". !?…")
    return _trim_to_sentence_start(stripped)


def _has_question_cue(candidate: str) -> bool:
    folded = candidate.casefold().lstrip("-—:;,. ")
    return any(
        folded == cue
        or folded.startswith(f"{cue} ")
        or f" {cue} " in f" {folded[:120]} "
        for cue in _QUESTION_CUES
    )


class QuestionDetector:
    """Detect completed interview questions from cumulative final transcripts."""

    def __init__(self) -> None:
        self._last_transcript = ""
        self._emitted: set[str] = set()

    def reset(self) -> None:
        self._last_transcript = ""
        self._emitted.clear()

    def feed(self, full_text: str) -> DetectedQuestion | None:
        transcript = _normalise(full_text)
        if not transcript or transcript == self._last_transcript:
            return None
        self._last_transcript = transcript

        explicit = _last_explicit_question(transcript)
        if explicit is not None:
            key = _dedup_key(explicit)
            if key and key not in self._emitted:
                self._emitted.add(key)
                return DetectedQuestion(
                    text=explicit,
                    confidence=0.98,
                    reason="question_mark",
                )

        candidate = _last_sentence(transcript)
        word_count = len(candidate.split())
        if not (4 <= word_count <= 60):
            return None
        if not (15 <= len(candidate) <= _MAX_QUESTION_CHARS):
            return None
        if not _has_question_cue(candidate):
            return None

        key = _dedup_key(candidate)
        if not key or key in self._emitted:
            return None
        self._emitted.add(key)
        return DetectedQuestion(
            text=candidate,
            confidence=0.74,
            reason="interrogative_cue",
        )
