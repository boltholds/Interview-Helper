from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Interview Helper API"
    app_env: str = "development"
    api_prefix: str = "/api/v1"
    backend_cors_origins: list[str] = ["http://localhost:5173"]
    knowledge_index_path: str = "data/index/knowledge.db"

    stt_provider: str = "stub"
    llm_provider: str = "stub"
    embedding_provider: str = "local-hash"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_dimensions: int = 256
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
