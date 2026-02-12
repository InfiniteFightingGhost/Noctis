from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260211_0002"
down_revision = "20260211_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("predictions", sa.Column("ground_truth_stage", sa.String(length=8)))
    op.create_index(
        "ix_predictions_model_version",
        "predictions",
        ["model_version", "window_end_ts"],
    )

    op.create_table(
        "evaluation_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("recording_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("from_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("to_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metrics", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_evaluation_metrics_model",
        "evaluation_metrics",
        ["model_version", "created_at"],
    )
    op.create_index(
        "ix_evaluation_metrics_recording",
        "evaluation_metrics",
        ["recording_id", "created_at"],
    )

    op.create_table(
        "model_usage_stats",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("window_start_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prediction_count", sa.Integer(), nullable=False),
        sa.Column("average_latency_ms", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_model_usage_stats_model",
        "model_usage_stats",
        ["model_version", "created_at"],
    )

    op.create_table(
        "feature_statistics",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "recording_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("recordings.id"),
            nullable=False,
        ),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("feature_schema_version", sa.String(length=64), nullable=False),
        sa.Column(
            "window_end_ts",
            sa.DateTime(timezone=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("stats", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_feature_statistics_recording_time",
        "feature_statistics",
        ["recording_id", "window_end_ts"],
    )
    op.create_index(
        "ix_feature_statistics_model_time",
        "feature_statistics",
        ["model_version", "window_end_ts"],
    )

    op.execute(
        "SELECT create_hypertable('feature_statistics', 'window_end_ts', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE)"
    )


def downgrade() -> None:
    op.drop_index("ix_feature_statistics_model_time", table_name="feature_statistics")
    op.drop_index(
        "ix_feature_statistics_recording_time", table_name="feature_statistics"
    )
    op.drop_table("feature_statistics")
    op.drop_index("ix_model_usage_stats_model", table_name="model_usage_stats")
    op.drop_table("model_usage_stats")
    op.drop_index("ix_evaluation_metrics_recording", table_name="evaluation_metrics")
    op.drop_index("ix_evaluation_metrics_model", table_name="evaluation_metrics")
    op.drop_table("evaluation_metrics")
    op.drop_index("ix_predictions_model_version", table_name="predictions")
    op.drop_column("predictions", "ground_truth_stage")
