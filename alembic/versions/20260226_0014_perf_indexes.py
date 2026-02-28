from __future__ import annotations

from alembic import op


revision = "20260226_0014"
down_revision = "20260224_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_epochs_recording_time")
    op.execute("DROP INDEX IF EXISTS ix_predictions_recording_time")

    op.create_index(
        "ix_recordings_tenant_started_at",
        "recordings",
        ["tenant_id", "started_at"],
    )
    op.create_index(
        "ix_devices_tenant_created_at",
        "devices",
        ["tenant_id", "created_at"],
    )
    op.create_index(
        "ix_alarms_tenant_created_at",
        "alarms",
        ["tenant_id", "created_at"],
    )
    op.create_index(
        "ix_routines_tenant_status_updated_at",
        "routines",
        ["tenant_id", "status", "updated_at"],
    )
    op.create_index(
        "ix_predictions_tenant_window_end_ts",
        "predictions",
        ["tenant_id", "window_end_ts"],
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_tenant_window_end_ts", table_name="predictions")
    op.drop_index("ix_routines_tenant_status_updated_at", table_name="routines")
    op.drop_index("ix_alarms_tenant_created_at", table_name="alarms")
    op.drop_index("ix_devices_tenant_created_at", table_name="devices")
    op.drop_index("ix_recordings_tenant_started_at", table_name="recordings")

    op.create_index(
        "ix_predictions_recording_time",
        "predictions",
        ["recording_id", "window_end_ts"],
    )
    op.create_index(
        "ix_epochs_recording_time",
        "epochs",
        ["recording_id", "epoch_start_ts"],
    )
