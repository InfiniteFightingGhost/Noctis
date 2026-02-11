from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Device(Base):
    __tablename__ = "devices"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200))
    external_id: Mapped[str | None] = mapped_column(String(200), unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    recordings: Mapped[list["Recording"]] = relationship(back_populates="device")


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
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
        Index("ix_epochs_recording_time", "recording_id", "epoch_start_ts"),
        Index("ix_epochs_recording_index", "recording_id", "epoch_index"),
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
        Index("ix_predictions_recording_time", "recording_id", "window_end_ts"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
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
    predicted_stage: Mapped[str] = mapped_column(String(8))
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
    details: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
