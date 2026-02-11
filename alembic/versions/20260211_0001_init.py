from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260211_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    op.create_table(
        "devices",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("external_id", sa.String(length=200), nullable=True, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "recordings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "device_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("devices.id"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("timezone", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "epochs",
        sa.Column(
            "recording_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("recordings.id"),
            primary_key=True,
        ),
        sa.Column("epoch_index", sa.Integer(), nullable=False),
        sa.Column(
            "epoch_start_ts",
            sa.DateTime(timezone=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("feature_schema_version", sa.String(length=64), nullable=False),
        sa.Column("features_payload", postgresql.JSONB(), nullable=False),
        sa.Column("ingest_ts", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_epochs_recording_time", "epochs", ["recording_id", "epoch_start_ts"]
    )
    op.create_index(
        "ix_epochs_recording_index", "epochs", ["recording_id", "epoch_index"]
    )
    op.create_index("ix_epochs_epoch_start_ts", "epochs", ["epoch_start_ts"])

    op.create_table(
        "predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "recording_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("recordings.id"),
            nullable=False,
        ),
        sa.Column("window_start_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "window_end_ts",
            sa.DateTime(timezone=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("feature_schema_version", sa.String(length=64), nullable=False),
        sa.Column("predicted_stage", sa.String(length=8), nullable=False),
        sa.Column("probabilities", postgresql.JSONB(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "recording_id", "window_end_ts", name="uq_prediction_window_end"
        ),
    )
    op.create_index(
        "ix_predictions_recording_time",
        "predictions",
        ["recording_id", "window_end_ts"],
    )
    op.create_index("ix_predictions_window_end_ts", "predictions", ["window_end_ts"])

    op.create_table(
        "model_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("version", sa.String(length=64), nullable=False, unique=True),
        sa.Column("details", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.execute(
        "SELECT create_hypertable('epochs', 'epoch_start_ts', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE)"
    )
    op.execute(
        "SELECT create_hypertable('predictions', 'window_end_ts', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE)"
    )


def downgrade() -> None:
    op.drop_table("model_versions")
    op.drop_table("predictions")
    op.drop_table("epochs")
    op.drop_table("recordings")
    op.drop_table("devices")
