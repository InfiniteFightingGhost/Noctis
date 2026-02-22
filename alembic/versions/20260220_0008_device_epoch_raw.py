from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260220_0008"
down_revision = "20260213_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "device_epoch_raw",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("device_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("recording_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("epoch_index", sa.Integer(), nullable=False),
        sa.Column("epoch_start_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_metrics", postgresql.JSONB(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["device_id"], ["devices.id"]),
        sa.ForeignKeyConstraint(["recording_id"], ["recordings.id"]),
        sa.UniqueConstraint(
            "tenant_id",
            "recording_id",
            "epoch_start_ts",
            name="uq_device_epoch_raw_recording_ts",
        ),
    )
    op.create_index(
        "ix_device_epoch_raw_tenant_recording",
        "device_epoch_raw",
        ["tenant_id", "recording_id"],
    )
    op.create_index(
        "ix_device_epoch_raw_tenant_device",
        "device_epoch_raw",
        ["tenant_id", "device_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_device_epoch_raw_tenant_device",
        table_name="device_epoch_raw",
    )
    op.drop_index(
        "ix_device_epoch_raw_tenant_recording",
        table_name="device_epoch_raw",
    )
    op.drop_table("device_epoch_raw")
