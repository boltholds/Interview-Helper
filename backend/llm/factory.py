from __future__ import annotations

from typing import TYPE_CHECKING

from llm.base import LLMProvider
from llm.openrouter import OpenRouterLLMProvider

if TYPE_CHECKING:
    from app.core.config import Settings


def create_llm_provider(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.casefold()
    if provider == "openrouter":
        return OpenRouterLLMProvider(
            api_key=settings.openrouter_api_key or "",
            model=settings.llm_model,
            base_url=settings.openrouter_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            http_referer=settings.openrouter_http_referer,
            app_title=settings.openrouter_app_title,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
