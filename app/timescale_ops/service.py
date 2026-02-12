from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from app.core.settings import get_settings


@dataclass(frozen=True)
class TimescaleAction:
    name: str
    sql: str


def fetch_policy_state(session) -> dict[str, Any]:
    retention = (
        session.execute(
            text(
                "SELECT hypertable_name, drop_after FROM timescaledb_information.policy_retention"
            )
        )
        .mappings()
        .all()
    )
    compression = (
        session.execute(
            text(
                "SELECT hypertable_name, compress_after FROM timescaledb_information.policy_compression"
            )
        )
        .mappings()
        .all()
    )
    aggregates = (
        session.execute(
            text(
                "SELECT view_name, materialization_hypertable_name, refresh_lag, schedule_interval "
                "FROM timescaledb_information.continuous_aggregates"
            )
        )
        .mappings()
        .all()
    )
    return {
        "retention": [dict(row) for row in retention],
        "compression": [dict(row) for row in compression],
        "continuous_aggregates": [dict(row) for row in aggregates],
    }


def build_policy_actions() -> list[TimescaleAction]:
    settings = get_settings()
    actions = [
        TimescaleAction(
            name="epochs_chunk_interval",
            sql=(
                "SELECT set_chunk_time_interval('epochs', "
                f"INTERVAL '{settings.timescale_chunk_interval_days} days')"
            ),
        ),
        TimescaleAction(
            name="predictions_chunk_interval",
            sql=(
                "SELECT set_chunk_time_interval('predictions', "
                f"INTERVAL '{settings.timescale_chunk_interval_days} days')"
            ),
        ),
        TimescaleAction(
            name="enable_epochs_compression",
            sql=(
                "ALTER TABLE epochs SET ("
                "timescaledb.compress, "
                f"timescaledb.compress_segmentby='{settings.timescale_compression_segmentby}'"
                ")"
            ),
        ),
        TimescaleAction(
            name="epochs_retention",
            sql=(
                "SELECT add_retention_policy('epochs', "
                f"INTERVAL '{settings.epochs_retention_days} days', if_not_exists => TRUE)"
            ),
        ),
        TimescaleAction(
            name="predictions_retention",
            sql=(
                "SELECT add_retention_policy('predictions', "
                f"INTERVAL '{settings.predictions_retention_days} days', if_not_exists => TRUE)"
            ),
        ),
        TimescaleAction(
            name="epochs_compression",
            sql=(
                "SELECT add_compression_policy('epochs', "
                f"INTERVAL '{settings.epochs_compression_after_days} days', if_not_exists => TRUE)"
            ),
        ),
    ]
    return actions


def apply_policy_actions(session) -> list[dict[str, str]]:
    applied: list[dict[str, str]] = []
    for action in build_policy_actions():
        session.execute(text(action.sql))
        applied.append({"name": action.name, "sql": action.sql})
    return applied
