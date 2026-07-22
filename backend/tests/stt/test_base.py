from stt.base import merge_transcript_text


def test_merge_transcript_text_removes_word_overlap() -> None:
    merged = merge_transcript_text(
        "We use idempotent workers with retries",
        "with retries and exponential backoff",
    )

    assert merged == "We use idempotent workers with retries and exponential backoff"


def test_merge_transcript_text_keeps_non_overlapping_text() -> None:
    merged = merge_transcript_text("First answer.", "Second answer.")

    assert merged == "First answer. Second answer."
