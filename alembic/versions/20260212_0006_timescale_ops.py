from __future__ import annotations

from alembic import op


revision = "20260212_0006"
down_revision = "20260212_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SELECT set_chunk_time_interval('epochs', INTERVAL '1 day')")
    op.execute("SELECT set_chunk_time_interval('predictions', INTERVAL '1 day')")
    op.execute(
        "ALTER TABLE epochs SET (timescaledb.compress, timescaledb.compress_segmentby='recording_id')"
    )
    op.execute(
        "SELECT add_retention_policy('epochs', INTERVAL '90 days', if_not_exists => TRUE)"
    )
    op.execute(
        "SELECT add_retention_policy('predictions', INTERVAL '180 days', if_not_exists => TRUE)"
    )
    op.execute(
        "SELECT add_compression_policy('epochs', INTERVAL '7 days', if_not_exists => TRUE)"
    )

    with op.get_context().autocommit_block():
        op.execute(
            "CREATE MATERIALIZED VIEW IF NOT EXISTS recording_daily_summary "
            "WITH (timescaledb.continuous) AS "
            "SELECT tenant_id, recording_id, "
            "time_bucket('1 day', window_end_ts) AS day, "
            "count(*) AS prediction_count, "
            "avg(confidence) AS avg_confidence "
            "FROM predictions "
            "GROUP BY tenant_id, recording_id, day"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_recording_daily_summary_tenant_day "
            "ON recording_daily_summary (tenant_id, day)"
        )

        op.execute(
            "CREATE MATERIALIZED VIEW IF NOT EXISTS device_daily_summary "
            "WITH (timescaledb.continuous) AS "
            "SELECT r.tenant_id, r.device_id, "
            "time_bucket('1 day', e.epoch_start_ts) AS day, "
            "count(*) AS epoch_count "
            "FROM epochs e "
            "JOIN recordings r ON e.recording_id = r.id AND e.tenant_id = r.tenant_id "
            "GROUP BY r.tenant_id, r.device_id, day"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_device_daily_summary_tenant_day "
            "ON device_daily_summary (tenant_id, day)"
        )

    op.execute(
        "SELECT add_continuous_aggregate_policy('recording_daily_summary', "
        "start_offset => INTERVAL '30 days', "
        "end_offset => INTERVAL '1 day', "
        "schedule_interval => INTERVAL '1 hour', if_not_exists => TRUE)"
    )
    op.execute(
        "SELECT add_continuous_aggregate_policy('device_daily_summary', "
        "start_offset => INTERVAL '30 days', "
        "end_offset => INTERVAL '1 day', "
        "schedule_interval => INTERVAL '1 hour', if_not_exists => TRUE)"
    )


def downgrade() -> None:
    op.execute(
        "SELECT remove_continuous_aggregate_policy('device_daily_summary', if_exists => TRUE)"
    )
    op.execute(
        "SELECT remove_continuous_aggregate_policy('recording_daily_summary', if_exists => TRUE)"
    )
    op.execute("DROP MATERIALIZED VIEW IF EXISTS device_daily_summary")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS recording_daily_summary")
    op.execute("SELECT remove_retention_policy('predictions', if_exists => TRUE)")
    op.execute("SELECT remove_retention_policy('epochs', if_exists => TRUE)")
    op.execute("SELECT remove_compression_policy('epochs', if_exists => TRUE)")
