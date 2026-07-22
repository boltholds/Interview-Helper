import math

from ingestion.embeddings import HashingEmbeddingProvider


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_hashing_embeddings_are_deterministic_and_normalized() -> None:
    provider = HashingEmbeddingProvider(dimensions=64)

    first = provider.embed_query("idempotent worker retries")
    second = provider.embed_query("idempotent worker retries")

    assert first == second
    assert len(first) == 64
    assert math.isclose(sum(value * value for value in first), 1.0, rel_tol=1e-6)


def test_hashing_embeddings_reward_shared_terms() -> None:
    provider = HashingEmbeddingProvider(dimensions=128)
    query = provider.embed_query("worker retries")
    related = provider.embed_query("retries for background worker")
    unrelated = provider.embed_query("react browser components")

    assert _cosine(query, related) > _cosine(query, unrelated)
