from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Interview Helper API"
    app_env: str = "development"
    api_prefix: str = "/api/v1"
    backend_cors_origins: list[str] = ["http://localhost:5173"]
    knowledge_index_path: str = "data/index/knowledge.db"

    stt_provider: str = "whispercpp"
    whispercpp_binary_path: str = "whisper-cli"
    whispercpp_model_path: str = "models/ggml-small.bin"
    whispercpp_language: str = "auto"
    whispercpp_threads: int = 6
    whispercpp_step_ms: int = 3_000
    whispercpp_window_ms: int = 12_000
    whispercpp_overlap_ms: int = 1_000
    whispercpp_minimum_audio_ms: int = 500
    whispercpp_timeout_seconds: float = 120.0
    whispercpp_use_gpu: bool = True
    whispercpp_flash_attention: bool = True

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
