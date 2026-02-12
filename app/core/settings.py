from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", protected_namespaces=()
    )

    app_name: str = "noctis-ml-api"
    env: str = "local"
    log_level: str = "INFO"

    api_v1_prefix: str = "/v1"
    auth_header: str = "Authorization"
    jwt_issuer: str = "noctis"
    jwt_audience: str = "noctis-services"
    jwt_allowed_algorithms: list[str] = ["RS256", "HS256"]
    jwt_leeway_seconds: int = 30
    default_tenant_id: str = "00000000-0000-0000-0000-000000000001"

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

    request_timeout_seconds: float = 30.0

    db_retry_max_attempts: int = 3
    db_retry_base_delay_seconds: float = 0.2
    db_retry_max_delay_seconds: float = 2.0
    db_circuit_failure_threshold: int = 5
    db_circuit_recovery_seconds: int = 30

    drift_psi_threshold: float = 0.2
    drift_kl_threshold: float = 0.1
    drift_z_threshold: float = 3.0
    retrain_drift_threshold: float = 1.0

    performance_sample_size: int = 250
    inference_batch_size: int = 64
    enable_batch_inference: bool = True

    epochs_retention_days: int = 90
    predictions_retention_days: int = 180
    epochs_compression_after_days: int = 7
    timescale_chunk_interval_days: int = 1
    timescale_compression_segmentby: str = "recording_id"

    audit_poll_interval_seconds: float = 900.0
    audit_max_report_rows: int = 500
    audit_epoch_gap_seconds: int = 30

    slo_inference_p95_ms: float = 350.0
    slo_inference_p99_ms: float = 700.0
    slo_ingest_failure_rate: float = 0.01
    slo_db_commit_p95_ms: float = 200.0
    slo_model_reload_success_rate: float = 0.99

    promotion_min_accuracy: float = 0.8
    promotion_min_macro_f1: float = 0.7
    promotion_block_if_missing_metrics: bool = True

    retrain_poll_interval_seconds: float = 60.0
    retrain_batch_size: int = 1
    retrain_dataset_output_root: Path = Path("data/retrain_datasets")
    retrain_model_output_root: Path = Path("models")
    retrain_feature_schema_path: Path = Path("models/active/feature_schema.json")
    retrain_experiment_name: str = "drift-retrain"

    backup_dir: Path = Path("backups")


@lru_cache
def get_settings() -> Settings:
    return Settings()
