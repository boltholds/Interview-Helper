from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Interview Helper API"
    app_env: str = "development"
    api_prefix: str = "/api/v1"
    backend_cors_origins: list[str] = ["http://localhost:5173"]

    stt_provider: str = "stub"
    llm_provider: str = "stub"
    embedding_provider: str = "stub"
    openai_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
