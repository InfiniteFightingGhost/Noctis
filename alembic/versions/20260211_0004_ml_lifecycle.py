from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260211_0004"
down_revision = "20260211_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_versions",
        sa.Column("status", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("metrics", sa.JSON(), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("feature_schema_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("artifact_path", sa.String(length=256), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("promoted_by", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute("UPDATE model_versions SET status = 'training' WHERE status IS NULL")
    op.alter_column("model_versions", "status", nullable=False)
    op.create_index(
        "uq_model_versions_production",
        "model_versions",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'production'"),
    )

    op.create_table(
        "experiments",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=128), nullable=False, unique=True),
        sa.Column("description", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "training_runs",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "experiment_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True
        ),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("hyperparameters", sa.JSON(), nullable=True),
        sa.Column("dataset_snapshot", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("feature_schema_version", sa.String(length=64), nullable=True),
        sa.Column("commit_hash", sa.String(length=64), nullable=True),
        sa.Column("artifact_path", sa.String(length=256), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_training_runs_model",
        "training_runs",
        ["model_version", "created_at"],
    )

    op.create_table(
        "model_promotion_events",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("previous_status", sa.String(length=32), nullable=True),
        sa.Column("new_status", sa.String(length=32), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("reason", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "retrain_jobs",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("drift_score", sa.Float(), nullable=False),
        sa.Column("triggering_features", sa.JSON(), nullable=True),
        sa.Column("suggested_from_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("suggested_to_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dataset_config", sa.JSON(), nullable=True),
        sa.Column("training_config", sa.JSON(), nullable=True),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.String(length=512), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_retrain_jobs_status",
        "retrain_jobs",
        ["status", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_retrain_jobs_status", table_name="retrain_jobs")
    op.drop_table("retrain_jobs")
    op.drop_table("model_promotion_events")
    op.drop_index("ix_training_runs_model", table_name="training_runs")
    op.drop_table("training_runs")
    op.drop_table("experiments")
    op.drop_index("uq_model_versions_production", table_name="model_versions")
    op.drop_column("model_versions", "archived_at")
    op.drop_column("model_versions", "promoted_by")
    op.drop_column("model_versions", "promoted_at")
    op.drop_column("model_versions", "artifact_path")
    op.drop_column("model_versions", "feature_schema_version")
    op.drop_column("model_versions", "metrics")
    op.drop_column("model_versions", "status")
