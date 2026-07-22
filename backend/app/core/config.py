from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Interview Helper API"
    app_env: str = "development"
    api_prefix: str = "/api/v1"
    backend_cors_origins: list[str] = ["http://localhost:5173"]
    knowledge_index_path: str = "data/index/knowledge.db"

    stt_provider: str = "whispercpp"
    whispercpp_base_url: str = "http://whispercpp:8080"
    whispercpp_language: str = "auto"
    whispercpp_step_ms: int = 3_000
    whispercpp_window_ms: int = 12_000
    whispercpp_overlap_ms: int = 1_000
    whispercpp_minimum_audio_ms: int = 500
    whispercpp_timeout_seconds: float = 120.0

    llm_provider: str = "openrouter"
    llm_model: str = "~google/gemini-flash-latest"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1_200

    embedding_provider: str = "openrouter"
    embedding_model: str = "qwen/qwen3-embedding-8b"
    local_embedding_dimensions: int = 256

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_http_referer: str | None = None
    openrouter_app_title: str = "Interview Helper"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
