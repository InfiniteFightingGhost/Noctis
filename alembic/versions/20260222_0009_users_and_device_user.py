from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260222_0009"
down_revision = "20260220_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("external_id", sa.String(length=200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("tenant_id", "external_id", name="uq_users_tenant_external"),
    )
    op.create_index("ix_users_tenant", "users", ["tenant_id"])

    op.add_column("devices", sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index("ix_devices_tenant_user", "devices", ["tenant_id", "user_id"])
    op.create_foreign_key(
        "fk_devices_user",
        "devices",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_devices_user", "devices", type_="foreignkey")
    op.drop_index("ix_devices_tenant_user", table_name="devices")
    op.drop_column("devices", "user_id")

    op.drop_index("ix_users_tenant", table_name="users")
    op.drop_table("users")
