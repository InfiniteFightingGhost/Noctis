from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260212_0005"
down_revision = "20260211_0004"
branch_labels = None
depends_on = None

DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001"


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("name", name="uq_tenants_name"),
    )
    op.execute(
        "INSERT INTO tenants (id, name, status, created_at) "
        f"VALUES ('{DEFAULT_TENANT_ID}', 'default', 'active', NOW())"
    )

    _add_tenant_column("devices")
    _add_tenant_column("recordings")
    _add_tenant_column("epochs")
    _add_tenant_column("predictions")
    _add_tenant_column("model_usage_stats")
    _add_tenant_column("experiments")
    _add_tenant_column("feature_statistics")
    _add_tenant_column("evaluation_metrics")
    _add_tenant_column("retrain_jobs")

    op.create_index("ix_devices_tenant", "devices", ["tenant_id"])
    op.create_index("ix_recordings_tenant", "recordings", ["tenant_id"])
    op.create_index(
        "ix_recordings_tenant_device", "recordings", ["tenant_id", "device_id"]
    )

    op.create_index(
        "ix_epochs_tenant_recording_time",
        "epochs",
        ["tenant_id", "recording_id", "epoch_start_ts"],
    )
    op.create_index(
        "ix_epochs_tenant_recording_index",
        "epochs",
        ["tenant_id", "recording_id", "epoch_index"],
    )
    op.create_index(
        "ix_epochs_tenant_epoch_start_ts",
        "epochs",
        ["tenant_id", "epoch_start_ts"],
    )

    op.create_index(
        "ix_predictions_tenant_recording_time",
        "predictions",
        ["tenant_id", "recording_id", "window_end_ts"],
    )
    op.create_index(
        "ix_predictions_tenant_model_version",
        "predictions",
        ["tenant_id", "model_version", "window_end_ts"],
    )

    op.create_index(
        "ix_model_usage_stats_tenant_model",
        "model_usage_stats",
        ["tenant_id", "model_version", "created_at"],
    )

    op.create_index("ix_experiments_tenant", "experiments", ["tenant_id"])
    op.create_index(
        "ix_feature_statistics_tenant_date",
        "feature_statistics",
        ["tenant_id", "stat_date"],
    )
    op.create_index(
        "ix_evaluation_metrics_tenant",
        "evaluation_metrics",
        ["tenant_id", "created_at"],
    )
    op.create_index(
        "ix_retrain_jobs_tenant_status",
        "retrain_jobs",
        ["tenant_id", "status", "created_at"],
    )

    op.drop_constraint("devices_external_id_key", "devices", type_="unique")
    op.create_unique_constraint(
        "uq_devices_tenant_external", "devices", ["tenant_id", "external_id"]
    )

    op.drop_constraint("experiments_name_key", "experiments", type_="unique")
    op.create_unique_constraint(
        "uq_experiments_tenant_name", "experiments", ["tenant_id", "name"]
    )

    op.drop_constraint(
        "uq_feature_statistics_daily", "feature_statistics", type_="unique"
    )
    op.create_unique_constraint(
        "uq_feature_statistics_daily",
        "feature_statistics",
        [
            "tenant_id",
            "recording_id",
            "model_version",
            "feature_schema_version",
            "stat_date",
            "window_end_ts",
        ],
    )

    op.create_table(
        "service_clients",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=128), nullable=False, unique=True),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_service_clients_tenant", "service_clients", ["tenant_id"])

    op.create_table(
        "service_client_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "client_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("service_clients.id"),
            nullable=False,
        ),
        sa.Column("key_id", sa.String(length=64), nullable=False),
        sa.Column("public_key", sa.Text(), nullable=True),
        sa.Column("secret", sa.String(length=256), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("key_id", name="uq_service_client_keys_key_id"),
    )
    op.create_index(
        "ix_service_client_keys_client",
        "service_client_keys",
        ["client_id", "status"],
    )

    op.create_table(
        "auditor_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("issue_type", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("recording_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_auditor_reports_tenant", "auditor_reports", ["tenant_id", "detected_at"]
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("target_type", sa.String(length=128), nullable=False),
        sa.Column("target_id", sa.String(length=128), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_logs_tenant", "audit_logs", ["tenant_id", "timestamp"])
    op.execute(
        "CREATE OR REPLACE FUNCTION audit_log_append_only() RETURNS trigger AS $$"
        "BEGIN RAISE EXCEPTION 'audit_logs is append-only'; END; $$ LANGUAGE plpgsql"
    )
    op.execute(
        "CREATE TRIGGER audit_logs_no_update "
        "BEFORE UPDATE OR DELETE ON audit_logs "
        "FOR EACH ROW EXECUTE FUNCTION audit_log_append_only()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS audit_logs_no_update ON audit_logs")
    op.execute("DROP FUNCTION IF EXISTS audit_log_append_only")
    op.drop_index("ix_audit_logs_tenant", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index("ix_auditor_reports_tenant", table_name="auditor_reports")
    op.drop_table("auditor_reports")

    op.drop_index("ix_service_client_keys_client", table_name="service_client_keys")
    op.drop_index("ix_service_clients_tenant", table_name="service_clients")
    op.drop_table("service_client_keys")
    op.drop_table("service_clients")

    op.drop_constraint(
        "uq_feature_statistics_daily", "feature_statistics", type_="unique"
    )
    op.create_unique_constraint(
        "uq_feature_statistics_daily",
        "feature_statistics",
        [
            "recording_id",
            "model_version",
            "feature_schema_version",
            "stat_date",
            "window_end_ts",
        ],
    )

    op.drop_constraint("uq_experiments_tenant_name", "experiments", type_="unique")
    op.create_unique_constraint("experiments_name_key", "experiments", ["name"])

    op.drop_constraint("uq_devices_tenant_external", "devices", type_="unique")
    op.create_unique_constraint("devices_external_id_key", "devices", ["external_id"])

    op.drop_index("ix_retrain_jobs_tenant_status", table_name="retrain_jobs")
    op.drop_index("ix_evaluation_metrics_tenant", table_name="evaluation_metrics")
    op.drop_index("ix_feature_statistics_tenant_date", table_name="feature_statistics")
    op.drop_index("ix_experiments_tenant", table_name="experiments")
    op.drop_index("ix_model_usage_stats_tenant_model", table_name="model_usage_stats")
    op.drop_index("ix_predictions_tenant_model_version", table_name="predictions")
    op.drop_index("ix_predictions_tenant_recording_time", table_name="predictions")
    op.drop_index("ix_epochs_tenant_epoch_start_ts", table_name="epochs")
    op.drop_index("ix_epochs_tenant_recording_index", table_name="epochs")
    op.drop_index("ix_epochs_tenant_recording_time", table_name="epochs")
    op.drop_index("ix_recordings_tenant_device", table_name="recordings")
    op.drop_index("ix_recordings_tenant", table_name="recordings")
    op.drop_index("ix_devices_tenant", table_name="devices")

    _drop_tenant_column("retrain_jobs")
    _drop_tenant_column("evaluation_metrics")
    _drop_tenant_column("feature_statistics")
    _drop_tenant_column("experiments")
    _drop_tenant_column("model_usage_stats")
    _drop_tenant_column("predictions")
    _drop_tenant_column("epochs")
    _drop_tenant_column("recordings")
    _drop_tenant_column("devices")

    op.drop_table("tenants")


def _add_tenant_column(table_name: str) -> None:
    op.add_column(
        table_name,
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
            server_default=DEFAULT_TENANT_ID,
        ),
    )
    op.execute(
        f"UPDATE {table_name} SET tenant_id = '{DEFAULT_TENANT_ID}' WHERE tenant_id IS NULL"
    )
    op.alter_column(table_name, "tenant_id", nullable=False, server_default=None)
    op.create_foreign_key(
        f"fk_{table_name}_tenant",
        table_name,
        "tenants",
        ["tenant_id"],
        ["id"],
    )


def _drop_tenant_column(table_name: str) -> None:
    op.drop_constraint(f"fk_{table_name}_tenant", table_name, type_="foreignkey")
    op.drop_column(table_name, "tenant_id")
