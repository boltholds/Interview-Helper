from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider cannot produce valid vectors."""


class EmbeddingProvider(ABC):
    name: str
    model: str

    @property
    @abstractmethod
    def dimensions(self) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        if len(vectors) != 1:
            raise EmbeddingProviderError("Embedding provider returned an invalid query response")
        return vectors[0]


def _normalise(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


class HashingEmbeddingProvider(EmbeddingProvider):
    """Small deterministic fallback for local development and tests."""

    name = "local-hash"

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 32:
            raise ValueError("Hashing embedding dimensions must be at least 32")
        self._dimensions = dimensions
        self.model = f"hashing-v1-{dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @staticmethod
    def _features(text: str) -> list[tuple[str, float]]:
        tokens = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
        features: list[tuple[str, float]] = [(f"w:{token}", 1.0) for token in tokens]
        compact = " ".join(tokens)
        padded = f"  {compact}  "
        features.extend(
            (f"c3:{padded[index:index + 3]}", 0.25)
            for index in range(max(0, len(padded) - 2))
        )
        return features

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self._dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign * weight
        return _normalise(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    name = "openai-compatible"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        batch_size: int = 64,
        timeout_seconds: float = 60.0,
        http_referer: str | None = None,
        app_title: str | None = None,
        document_input_type: str | None = None,
        query_input_type: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("An API key is required for the embedding provider")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.http_referer = http_referer
        self.app_title = app_title
        self.document_input_type = document_input_type
        self.query_input_type = query_input_type
        self._dimensions: int | None = None

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Interview-Helper/0.5",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        return headers

    def _request(
        self,
        texts: Sequence[str],
        *,
        input_type: str | None = None,
    ) -> list[list[float]]:
        body: dict[str, Any] = {
            "model": self.model,
            "input": list(texts),
            "encoding_format": "float",
        }
        if input_type:
            body["input_type"] = input_type

        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload: dict[str, Any] = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EmbeddingProviderError(
                f"{self.name} embeddings request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EmbeddingProviderError(
                f"{self.name} embeddings request failed: {exc}"
            ) from exc

        data = payload.get("data")
        if not isinstance(data, list):
            raise EmbeddingProviderError(
                f"{self.name} embeddings response does not contain data"
            )

        ordered = sorted(data, key=lambda item: item.get("index", -1))
        vectors: list[list[float]] = []
        for item in ordered:
            raw_vector = item.get("embedding")
            if not isinstance(raw_vector, list) or not raw_vector:
                raise EmbeddingProviderError(
                    f"{self.name} embeddings response contains an invalid vector"
                )
            vectors.append([float(value) for value in raw_vector])

        if len(vectors) != len(texts):
            raise EmbeddingProviderError(
                f"{self.name} returned {len(vectors)} embeddings for {len(texts)} inputs"
            )
        dimensions = len(vectors[0])
        if any(len(vector) != dimensions for vector in vectors):
            raise EmbeddingProviderError(
                f"{self.name} returned vectors with inconsistent dimensions"
            )
        self._dimensions = dimensions
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(
                self._request(
                    texts[start:start + self.batch_size],
                    input_type=self.document_input_type,
                )
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self._request([text], input_type=self.query_input_type)
        return vectors[0]


class OpenAIEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 64,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            batch_size=batch_size,
            timeout_seconds=timeout_seconds,
        )


class OpenRouterEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    name = "openrouter"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen/qwen3-embedding-8b",
        base_url: str = "https://openrouter.ai/api/v1",
        batch_size: int = 64,
        timeout_seconds: float = 60.0,
        http_referer: str | None = None,
        app_title: str = "Interview Helper",
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            batch_size=batch_size,
            timeout_seconds=timeout_seconds,
            http_referer=http_referer,
            app_title=app_title,
            document_input_type="search_document",
            query_input_type="search_query",
        )


def create_embedding_provider(
    provider: str | None = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    local_dimensions: int = 256,
    http_referer: str | None = None,
    app_title: str | None = None,
) -> EmbeddingProvider:
    selected = (provider or os.getenv("EMBEDDING_PROVIDER") or "local-hash").casefold()
    if selected in {"local-hash", "hash", "stub"}:
        return HashingEmbeddingProvider(dimensions=local_dimensions)
    if selected == "openrouter":
        return OpenRouterEmbeddingProvider(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY", ""),
            model=model or os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b"),
            base_url=base_url
            or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            http_referer=http_referer or os.getenv("OPENROUTER_HTTP_REFERER"),
            app_title=app_title
            or os.getenv("OPENROUTER_APP_TITLE", "Interview Helper"),
        )
    if selected == "openai":
        return OpenAIEmbeddingProvider(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            model=model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")
