from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260224_0013"
down_revision = "20260224_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "device_pairing_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "device_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("devices.id"),
            nullable=False,
        ),
        sa.Column("pairing_code", sa.String(length=12), nullable=False),
        sa.Column(
            "created_by_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "claimed_by_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_device_pairing_sessions_tenant_expires",
        "device_pairing_sessions",
        ["tenant_id", "expires_at"],
    )
    op.create_index(
        "ix_device_pairing_sessions_tenant_code",
        "device_pairing_sessions",
        ["tenant_id", "pairing_code"],
    )


def downgrade() -> None:
    op.drop_index("ix_device_pairing_sessions_tenant_code", table_name="device_pairing_sessions")
    op.drop_index("ix_device_pairing_sessions_tenant_expires", table_name="device_pairing_sessions")
    op.drop_table("device_pairing_sessions")
