from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"
    __table_args__ = (UniqueConstraint("name", name="uq_tenants_name"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class Device(Base):
    __tablename__ = "devices"
    __table_args__ = (
        UniqueConstraint("tenant_id", "external_id", name="uq_devices_tenant_external"),
        Index("ix_devices_tenant", "tenant_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id")
    )
    name: Mapped[str] = mapped_column(String(200))
    external_id: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    recordings: Mapped[list["Recording"]] = relationship(back_populates="device")


class Recording(Base):
    __tablename__ = "recordings"
    __table_args__ = (
        Index("ix_recordings_tenant_device", "tenant_id", "device_id"),
        Index("ix_recordings_tenant", "tenant_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id")
    )
    device_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("devices.id")
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    timezone: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    device: Mapped["Device"] = relationship(back_populates="recordings")
    epochs: Mapped[list["Epoch"]] = relationship(back_populates="recording")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="recording")


class Epoch(Base):
    __tablename__ = "epochs"
    __table_args__ = (
        Index(
            "ix_epochs_tenant_recording_time",
            "tenant_id",
            "recording_id",
            "epoch_start_ts",
        ),
        Index(
            "ix_epochs_tenant_recording_index",
            "tenant_id",
            "recording_id",
            "epoch_index",
        ),
        Index("ix_epochs_tenant_epoch_start_ts", "tenant_id", "epoch_start_ts"),
        Index("ix_epochs_recording_time", "recording_id", "epoch_start_ts"),
        Index("ix_epochs_recording_index", "recording_id", "epoch_index"),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id")
    )
    recording_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recordings.id"), primary_key=True
    )
    epoch_index: Mapped[int] = mapped_column()
    epoch_start_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, primary_key=True
    )
    feature_schema_version: Mapped[str] = mapped_column(String(64))
    features_payload: Mapped[dict] = mapped_column(JSONB)
    ingest_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    recording: Mapped["Recording"] = relationship(back_populates="epochs")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "recording_id", "window_end_ts", name="uq_prediction_window_end"
        ),
        Index(
            "ix_predictions_tenant_recording_time",
            "tenant_id",
            "recording_id",
            "window_end_ts",
        ),
        Index(
            "ix_predictions_tenant_model_version",
            "tenant_id",
            "model_version",
            "window_end_ts",
        ),
        Index("ix_predictions_recording_time", "recording_id", "window_end_ts"),
        Index("ix_predictions_model_version", "model_version", "window_end_ts"),
        Index("ix_predictions_snapshot", "dataset_snapshot_id", "window_end_ts"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id")
    )
    recording_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recordings.id")
    )
    window_start_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, primary_key=True
    )
    model_version: Mapped[str] = mapped_column(String(64))
    feature_schema_version: Mapped[str] = mapped_column(String(64))
    dataset_snapshot_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_snapshots.id")
    )
    predicted_stage: Mapped[str] = mapped_column(String(8))
    ground_truth_stage: Mapped[str | None] = mapped_column(String(8))
    probabilities: Mapped[dict] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    recording: Mapped["Recording"] = relationship(back_populates="predictions")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    version: Mapped[str] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(32), default="training")
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    feature_schema_version: Mapped[str | None] = mapped_column(String(64))
    dataset_snapshot_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_snapshots.id")
    )
    training_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("training_runs.id")
    )
    git_commit_hash: Mapped[str | None] = mapped_column(String(64))
    training_seed: Mapped[int | None] = mapped_column(Integer)
    metrics_hash: Mapped[str | None] = mapped_column(String(128))
    artifact_hash: Mapped[str | None] = mapped_column(String(128))
    artifact_path: Mapped[str | None] = mapped_column(String(256))
    details: Mapped[dict | None] = mapped_column(JSONB)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    promoted_by: Mapped[str | None] = mapped_column(String(128))
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deployed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class FeatureSchema(Base):
    __tablename__ = "feature_schemas"
    __table_args__ = (
        UniqueConstraint("version", name="uq_feature_schemas_version"),
        UniqueConstraint("hash", name="uq_feature_schemas_hash"),
        Index("ix_feature_schemas_active", "is_active"),
        Index("ix_feature_schemas_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    version: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    features: Mapped[list["FeatureSchemaFeature"]] = relationship(
        back_populates="schema", order_by="FeatureSchemaFeature.position"
    )


class FeatureSchemaFeature(Base):
    __tablename__ = "feature_schema_features"
    __table_args__ = (
        UniqueConstraint(
            "feature_schema_id",
            "name",
            name="uq_feature_schema_features_schema_name",
        ),
        UniqueConstraint(
            "feature_schema_id",
            "position",
            name="uq_feature_schema_features_schema_position",
        ),
        CheckConstraint("position >= 0", name="ck_feature_schema_features_position"),
        Index("ix_feature_schema_features_schema", "feature_schema_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    feature_schema_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("feature_schemas.id")
    )
    name: Mapped[str] = mapped_column(String(128))
    dtype: Mapped[str] = mapped_column(String(64))
    allowed_range: Mapped[dict | None] = mapped_column(JSONB)
    description: Mapped[str | None] = mapped_column(Text)
    introduced_in_version: Mapped[str | None] = mapped_column(String(64))
    deprecated_in_version: Mapped[str | None] = mapped_column(String(64))
    position: Mapped[int] = mapped_column(Integer)

    schema: Mapped["FeatureSchema"] = relationship(back_populates="features")


class DatasetSnapshot(Base):
    __tablename__ = "dataset_snapshots"
    __table_args__ = (
        UniqueConstraint("checksum", name="uq_dataset_snapshots_checksum"),
        CheckConstraint("row_count >= 0", name="ck_dataset_snapshots_row_count"),
        Index("ix_dataset_snapshots_feature_schema", "feature_schema_version"),
        Index("ix_dataset_snapshots_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(128))
    feature_schema_version: Mapped[str] = mapped_column(String(64))
    date_range_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    date_range_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    recording_filter: Mapped[dict | None] = mapped_column(JSONB)
    label_source: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    checksum: Mapped[str] = mapped_column(String(128))
    row_count: Mapped[int] = mapped_column(Integer)

    windows: Mapped[list["DatasetSnapshotWindow"]] = relationship(
        back_populates="snapshot", order_by="DatasetSnapshotWindow.window_order"
    )


class DatasetSnapshotWindow(Base):
    __tablename__ = "dataset_snapshot_windows"
    __table_args__ = (
        CheckConstraint("window_order >= 0", name="ck_dataset_snapshot_windows_order"),
        Index("ix_dataset_snapshot_windows_snapshot", "dataset_snapshot_id"),
        Index("ix_dataset_snapshot_windows_window", "recording_id", "window_end_ts"),
    )

    dataset_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dataset_snapshots.id"),
        primary_key=True,
    )
    window_order: Mapped[int] = mapped_column(Integer, primary_key=True)
    recording_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recordings.id")
    )
    window_end_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    label_value: Mapped[str | None] = mapped_column(String(64))
    label_source: Mapped[str | None] = mapped_column(String(32))

    snapshot: Mapped["DatasetSnapshot"] = relationship(back_populates="windows")


class Experiment(Base):
    __tablename__ = "experiments"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_experiments_tenant_name"),
        Index("ix_experiments_tenant", "tenant_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id")
    )
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class TrainingRun(Base):
    __tablename__ = "training_runs"
    __table_args__ = (Index("ix_training_runs_model", "model_version", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    experiment_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    model_version: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    hyperparameters: Mapped[dict | None] = mapped_column(JSONB)
    dataset_snapshot: Mapped[dict | None] = mapped_column(JSONB)
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    feature_schema_version: Mapped[str | None] = mapped_column(String(64))
    commit_hash: Mapped[str | None] = mapped_column(String(64))
    artifact_path: Mapped[str | None] = mapped_column(String(256))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ModelPromotionEvent(Base):
    __tablename__ = "model_promotion_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_version: Mapped[str] = mapped_column(String(64))
    previous_status: Mapped[str | None] = mapped_column(String(32))
    new_status: Mapped[str] = mapped_column(String(32))
    actor: Mapped[str] = mapped_column(String(128))
    reason: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class RetrainJob(Base):
    __tablename__ = "retrain_jobs"
    __table_args__ = (
        Index("ix_retrain_jobs_status", "status", "created_at"),
        Index("ix_retrain_jobs_tenant_status", "tenant_id", "status", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id")
    )
    status: Mapped[str] = mapped_column(String(32), default="pending")
    drift_score: Mapped[float] = mapped_column(Float)
    triggering_features: Mapped[dict | None] = mapped_column(JSONB)
    suggested_from_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    suggested_to_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    dataset_config: Mapped[dict | None] = mapped_column(JSONB)
    training_config: Mapped[dict | None] = mapped_column(JSONB)
    model_version: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(String(512))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"
    __table_args__ = (
        Index("ix_evaluation_metrics_model", "model_version", "created_at"),
        Index("ix_evaluation_metrics_recording", "recording_id", "created_at"),
        Index("ix_evaluation_metrics_tenant", "tenant_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id")
    )
    model_version: Mapped[str] = mapped_column(String(64))
    recording_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    from_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    to_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metrics: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ModelUsageStat(Base):
    __tablename__ = "model_usage_stats"
    __table_args__ = (
        Index("ix_model_usage_stats_model", "model_version", "created_at"),
        Index(
            "ix_model_usage_stats_tenant_model",
            "tenant_id",
            "model_version",
            "created_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id")
    )
    model_version: Mapped[str] = mapped_column(String(64))
    window_start_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    prediction_count: Mapped[int] = mapped_column(Integer)
    average_latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class FeatureStatistic(Base):
    __tablename__ = "feature_statistics"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "recording_id",
            "model_version",
            "feature_schema_version",
            "stat_date",
            "window_end_ts",
            name="uq_feature_statistics_daily",
        ),
        Index("ix_feature_statistics_recording_date", "recording_id", "stat_date"),
        Index("ix_feature_statistics_model_date", "model_version", "stat_date"),
        Index("ix_feature_statistics_tenant_date", "tenant_id", "stat_date"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id")
    )
    recording_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recordings.id")
    )
    model_version: Mapped[str] = mapped_column(String(64))
    feature_schema_version: Mapped[str] = mapped_column(String(64))
    stat_date: Mapped[date] = mapped_column(Date, index=True)
    window_end_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, index=True
    )
    stats: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ServiceClient(Base):
    __tablename__ = "service_clients"
    __table_args__ = (Index("ix_service_clients_tenant", "tenant_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id")
    )
    name: Mapped[str] = mapped_column(String(128), unique=True)
    role: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    keys: Mapped[list["ServiceClientKey"]] = relationship(back_populates="client")


class ServiceClientKey(Base):
    __tablename__ = "service_client_keys"
    __table_args__ = (
        UniqueConstraint("key_id", name="uq_service_client_keys_key_id"),
        Index("ix_service_client_keys_client", "client_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    client_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("service_clients.id")
    )
    key_id: Mapped[str] = mapped_column(String(64))
    public_key: Mapped[str | None] = mapped_column(Text)
    secret: Mapped[str | None] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    client: Mapped["ServiceClient"] = relationship(back_populates="keys")


class AuditorReport(Base):
    __tablename__ = "auditor_reports"
    __table_args__ = (Index("ix_auditor_reports_tenant", "tenant_id", "detected_at"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id")
    )
    issue_type: Mapped[str] = mapped_column(String(64))
    severity: Mapped[str] = mapped_column(String(16))
    recording_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_logs_tenant", "tenant_id", "timestamp"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id")
    )
    actor: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(128))
    target_type: Mapped[str] = mapped_column(String(128))
    target_id: Mapped[str | None] = mapped_column(String(128))
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSONB)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
