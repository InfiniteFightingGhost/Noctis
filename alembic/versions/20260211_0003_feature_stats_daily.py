from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260211_0003"
down_revision = "20260211_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DELETE FROM feature_statistics")
    op.add_column(
        "feature_statistics",
        sa.Column("stat_date", sa.Date(), nullable=True),
    )
    op.execute("UPDATE feature_statistics SET stat_date = DATE(window_end_ts)")
    op.execute(
        "UPDATE feature_statistics SET window_end_ts = date_trunc('day', window_end_ts)"
    )
    op.alter_column("feature_statistics", "stat_date", nullable=False)
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
    op.create_index(
        "ix_feature_statistics_recording_date",
        "feature_statistics",
        ["recording_id", "stat_date"],
    )
    op.create_index(
        "ix_feature_statistics_model_date",
        "feature_statistics",
        ["model_version", "stat_date"],
    )


def downgrade() -> None:
    op.drop_index("ix_feature_statistics_model_date", table_name="feature_statistics")
    op.drop_index(
        "ix_feature_statistics_recording_date", table_name="feature_statistics"
    )
    op.drop_constraint(
        "uq_feature_statistics_daily",
        "feature_statistics",
        type_="unique",
    )
    op.drop_column("feature_statistics", "stat_date")
