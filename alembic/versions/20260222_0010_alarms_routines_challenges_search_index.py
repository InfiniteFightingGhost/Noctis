from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260222_0010"
down_revision = "20260222_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "alarms",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_alarms_tenant", "alarms", ["tenant_id"])
    op.create_index("ix_alarms_tenant_scheduled", "alarms", ["tenant_id", "scheduled_for"])

    op.create_table(
        "routines",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_routines_tenant", "routines", ["tenant_id"])
    op.create_index("ix_routines_tenant_status", "routines", ["tenant_id", "status"])

    op.create_table(
        "challenges",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ends_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "progress >= 0.0 AND progress <= 1.0",
            name="ck_challenges_progress",
        ),
    )
    op.create_index("ix_challenges_tenant", "challenges", ["tenant_id"])
    op.create_index(
        "ix_challenges_tenant_status",
        "challenges",
        ["tenant_id", "status"],
    )
    op.create_index(
        "ix_challenges_tenant_window",
        "challenges",
        ["tenant_id", "starts_at", "ends_at"],
    )

    op.create_table(
        "search_index_entries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("subtitle", sa.Text(), nullable=True),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "tenant_id",
            "entity_type",
            "entity_id",
            name="uq_search_index_entries_entity",
        ),
    )
    op.create_index("ix_search_index_entries_tenant", "search_index_entries", ["tenant_id"])
    op.create_index(
        "ix_search_index_entries_tenant_type",
        "search_index_entries",
        ["tenant_id", "entity_type"],
    )
    op.create_index(
        "ix_search_index_entries_entity",
        "search_index_entries",
        ["entity_type", "entity_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_search_index_entries_entity", table_name="search_index_entries")
    op.drop_index("ix_search_index_entries_tenant_type", table_name="search_index_entries")
    op.drop_index("ix_search_index_entries_tenant", table_name="search_index_entries")
    op.drop_table("search_index_entries")

    op.drop_index("ix_challenges_tenant_window", table_name="challenges")
    op.drop_index("ix_challenges_tenant_status", table_name="challenges")
    op.drop_index("ix_challenges_tenant", table_name="challenges")
    op.drop_table("challenges")

    op.drop_index("ix_routines_tenant_status", table_name="routines")
    op.drop_index("ix_routines_tenant", table_name="routines")
    op.drop_table("routines")

    op.drop_index("ix_alarms_tenant_scheduled", table_name="alarms")
    op.drop_index("ix_alarms_tenant", table_name="alarms")
    op.drop_table("alarms")
