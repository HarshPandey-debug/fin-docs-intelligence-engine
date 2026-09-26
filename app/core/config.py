"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "FinDocs Compliance Intelligence Engine"
    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    app_api_key: str | None = None
    rate_limit_per_minute: int = Field(default=120, ge=1, le=10_000)
    qdrant_url: HttpUrl = "http://qdrant:6333"
    qdrant_collection: str = "findocs_documents"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = Field(default=768, ge=64, le=3072)
    generation_model: str = "gemini-2.0-flash"
    retrieval_limit: int = Field(default=8, ge=1, le=25)
    gemini_api_key: str | None = None
    llama_cloud_api_key: str | None = None
    document_storage_dir: Path = Path("data/uploads")
    max_upload_bytes: int = Field(default=52_428_800, ge=1_024, le=1_073_741_824)
    database_url: str = "sqlite:///./data/findocs.db"
    code_execution_timeout_seconds: int = Field(default=3, ge=1, le=30)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the immutable, process-wide settings instance."""

    return Settings()
