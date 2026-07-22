from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from llm.base import ChatMessage, LLMProvider


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter cannot produce a chat completion."""


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "".join(parts)


class OpenRouterLLMProvider(LLMProvider):
    name = "openrouter"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "~google/gemini-flash-latest",
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.2,
        max_tokens: int = 1_200,
        timeout_seconds: float = 120.0,
        http_referer: str | None = None,
        app_title: str = "Interview Helper",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for the OpenRouter LLM provider")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.http_referer = http_referer
        self.app_title = app_title
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds),
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        return headers

    @staticmethod
    def _parse_event(raw_data: str) -> str:
        try:
            payload: dict[str, Any] = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise OpenRouterError(f"OpenRouter returned invalid SSE JSON: {raw_data}") from exc

        error = payload.get("error")
        if error:
            if isinstance(error, dict):
                message = str(error.get("message", error))
            else:
                message = str(error)
            raise OpenRouterError(f"OpenRouter streaming error: {message}")

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        choice = choices[0]
        if not isinstance(choice, dict):
            return ""
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return ""
        return _content_text(delta.get("content"))

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        if not messages:
            raise ValueError("At least one chat message is required")

        request_body = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                headers=self._headers(),
                json=request_body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    text = self._parse_event(data)
                    if text:
                        yield text
        except httpx.HTTPStatusError as exc:
            detail = (await exc.response.aread()).decode("utf-8", errors="replace")
            raise OpenRouterError(
                f"OpenRouter chat request failed with HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"OpenRouter chat request failed: {exc}") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
