from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260213_0007"
down_revision = "20260212_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "feature_schemas",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("hash", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("version", name="uq_feature_schemas_version"),
        sa.UniqueConstraint("hash", name="uq_feature_schemas_hash"),
    )
    op.create_index(
        "ix_feature_schemas_active",
        "feature_schemas",
        ["is_active"],
    )
    op.create_index(
        "uq_feature_schemas_active",
        "feature_schemas",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active = true"),
    )
    op.create_index(
        "ix_feature_schemas_created_at",
        "feature_schemas",
        ["created_at"],
    )

    op.create_table(
        "feature_schema_features",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "feature_schema_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("feature_schemas.id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("dtype", sa.String(length=64), nullable=False),
        sa.Column("allowed_range", postgresql.JSONB(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("introduced_in_version", sa.String(length=64), nullable=True),
        sa.Column("deprecated_in_version", sa.String(length=64), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.CheckConstraint("position >= 0", name="ck_feature_schema_features_position"),
        sa.UniqueConstraint(
            "feature_schema_id",
            "name",
            name="uq_feature_schema_features_schema_name",
        ),
        sa.UniqueConstraint(
            "feature_schema_id",
            "position",
            name="uq_feature_schema_features_schema_position",
        ),
    )
    op.create_index(
        "ix_feature_schema_features_schema",
        "feature_schema_features",
        ["feature_schema_id"],
    )

    op.create_table(
        "dataset_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("feature_schema_version", sa.String(length=64), nullable=False),
        sa.Column("date_range_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("date_range_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recording_filter", postgresql.JSONB(), nullable=True),
        sa.Column("label_source", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("checksum", sa.String(length=128), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.CheckConstraint("row_count >= 0", name="ck_dataset_snapshots_row_count"),
        sa.UniqueConstraint("checksum", name="uq_dataset_snapshots_checksum"),
    )
    op.create_index(
        "ix_dataset_snapshots_feature_schema",
        "dataset_snapshots",
        ["feature_schema_version"],
    )
    op.create_index(
        "ix_dataset_snapshots_created_at",
        "dataset_snapshots",
        ["created_at"],
    )

    op.create_table(
        "dataset_snapshot_windows",
        sa.Column(
            "dataset_snapshot_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("dataset_snapshots.id"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("window_order", sa.Integer(), primary_key=True, nullable=False),
        sa.Column(
            "recording_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("recordings.id"),
            nullable=False,
        ),
        sa.Column("window_end_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label_value", sa.String(length=64), nullable=True),
        sa.Column("label_source", sa.String(length=32), nullable=True),
        sa.CheckConstraint(
            "window_order >= 0", name="ck_dataset_snapshot_windows_order"
        ),
    )
    op.create_index(
        "ix_dataset_snapshot_windows_snapshot",
        "dataset_snapshot_windows",
        ["dataset_snapshot_id"],
    )
    op.create_index(
        "ix_dataset_snapshot_windows_window",
        "dataset_snapshot_windows",
        ["recording_id", "window_end_ts"],
    )

    op.add_column(
        "predictions",
        sa.Column("dataset_snapshot_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        "ix_predictions_snapshot",
        "predictions",
        ["dataset_snapshot_id", "window_end_ts"],
    )
    op.create_foreign_key(
        "fk_predictions_dataset_snapshot",
        "predictions",
        "dataset_snapshots",
        ["dataset_snapshot_id"],
        ["id"],
    )

    op.add_column(
        "model_versions",
        sa.Column("dataset_snapshot_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("training_run_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("git_commit_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("training_seed", sa.Integer(), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("metrics_hash", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("artifact_hash", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("deployed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_model_versions_dataset_snapshot",
        "model_versions",
        ["dataset_snapshot_id"],
    )
    op.create_index(
        "ix_model_versions_training_run",
        "model_versions",
        ["training_run_id"],
    )
    op.create_index(
        "ix_model_versions_deployed_at",
        "model_versions",
        ["deployed_at"],
    )
    op.create_foreign_key(
        "fk_model_versions_dataset_snapshot",
        "model_versions",
        "dataset_snapshots",
        ["dataset_snapshot_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_model_versions_training_run",
        "model_versions",
        "training_runs",
        ["training_run_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_model_versions_training_run",
        "model_versions",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_model_versions_dataset_snapshot",
        "model_versions",
        type_="foreignkey",
    )
    op.drop_index("ix_model_versions_deployed_at", table_name="model_versions")
    op.drop_index("ix_model_versions_training_run", table_name="model_versions")
    op.drop_index("ix_model_versions_dataset_snapshot", table_name="model_versions")
    op.drop_column("model_versions", "deployed_at")
    op.drop_column("model_versions", "artifact_hash")
    op.drop_column("model_versions", "metrics_hash")
    op.drop_column("model_versions", "training_seed")
    op.drop_column("model_versions", "git_commit_hash")
    op.drop_column("model_versions", "training_run_id")
    op.drop_column("model_versions", "dataset_snapshot_id")

    op.drop_constraint(
        "fk_predictions_dataset_snapshot",
        "predictions",
        type_="foreignkey",
    )
    op.drop_index("ix_predictions_snapshot", table_name="predictions")
    op.drop_column("predictions", "dataset_snapshot_id")

    op.drop_index(
        "ix_dataset_snapshot_windows_window",
        table_name="dataset_snapshot_windows",
    )
    op.drop_index(
        "ix_dataset_snapshot_windows_snapshot",
        table_name="dataset_snapshot_windows",
    )
    op.drop_table("dataset_snapshot_windows")
    op.drop_index(
        "ix_dataset_snapshots_created_at",
        table_name="dataset_snapshots",
    )
    op.drop_index(
        "ix_dataset_snapshots_feature_schema",
        table_name="dataset_snapshots",
    )
    op.drop_table("dataset_snapshots")
    op.drop_index(
        "ix_feature_schema_features_schema",
        table_name="feature_schema_features",
    )
    op.drop_table("feature_schema_features")
    op.drop_index(
        "ix_feature_schemas_created_at",
        table_name="feature_schemas",
    )
    op.drop_index("uq_feature_schemas_active", table_name="feature_schemas")
    op.drop_index("ix_feature_schemas_active", table_name="feature_schemas")
    op.drop_table("feature_schemas")
