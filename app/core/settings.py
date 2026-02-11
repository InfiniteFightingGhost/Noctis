from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", protected_namespaces=()
    )

    app_name: str = "noctis-ml-api"
    env: str = "local"
    log_level: str = "INFO"

    api_v1_prefix: str = "/v1"
    api_key_header: str = "X-API-Key"
    admin_key_header: str = "X-Admin-Key"
    api_key: str = "changeme"
    admin_key: str = "adminchangeme"

    cors_allow_origins: list[str] = ["*"]
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    database_url: str = "postgresql+psycopg://postgres:postgres@timescaledb:5432/noctis"
    db_pool_size: int = 5
    db_max_overflow: int = 10

    model_registry_path: Path = Path("models")
    active_model_version: str = "active"
    feature_schema_version: str = "v1"
    window_size: int = 21
    allow_window_padding: bool = False

    metrics_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
