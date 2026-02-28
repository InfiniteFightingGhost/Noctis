from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260224_0012"
down_revision = "20260224_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("alarms", sa.Column("wake_time", sa.String(length=5), nullable=True))
    op.add_column("alarms", sa.Column("wake_window_minutes", sa.Integer(), nullable=True))
    op.add_column("alarms", sa.Column("sunrise_enabled", sa.Boolean(), nullable=True))
    op.add_column("alarms", sa.Column("sunrise_intensity", sa.Integer(), nullable=True))
    op.add_column("alarms", sa.Column("sound_id", sa.String(length=64), nullable=True))
    op.add_column(
        "alarms",
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.execute("UPDATE alarms SET wake_time = '06:45' WHERE wake_time IS NULL")
    op.execute("UPDATE alarms SET wake_window_minutes = 20 WHERE wake_window_minutes IS NULL")
    op.execute("UPDATE alarms SET sunrise_enabled = true WHERE sunrise_enabled IS NULL")
    op.execute("UPDATE alarms SET sunrise_intensity = 3 WHERE sunrise_intensity IS NULL")
    op.execute("UPDATE alarms SET sound_id = 'ocean' WHERE sound_id IS NULL")

    op.alter_column("alarms", "wake_time", nullable=False)
    op.alter_column("alarms", "wake_window_minutes", nullable=False)
    op.alter_column("alarms", "sunrise_enabled", nullable=False)
    op.alter_column("alarms", "sunrise_intensity", nullable=False)
    op.alter_column("alarms", "sound_id", nullable=False)

    op.add_column(
        "routines",
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_table(
        "routine_steps",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "routine_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("routines.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=128), nullable=False),
        sa.Column("duration_minutes", sa.Integer(), nullable=False),
        sa.Column("emoji", sa.String(length=8), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_routine_steps_routine", "routine_steps", ["routine_id", "position"])
    op.create_index("ix_routine_steps_tenant", "routine_steps", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("ix_routine_steps_tenant", table_name="routine_steps")
    op.drop_index("ix_routine_steps_routine", table_name="routine_steps")
    op.drop_table("routine_steps")

    op.drop_column("routines", "updated_at")

    op.drop_column("alarms", "updated_at")
    op.drop_column("alarms", "sound_id")
    op.drop_column("alarms", "sunrise_intensity")
    op.drop_column("alarms", "sunrise_enabled")
    op.drop_column("alarms", "wake_window_minutes")
    op.drop_column("alarms", "wake_time")
