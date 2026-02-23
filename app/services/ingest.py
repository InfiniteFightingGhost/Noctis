from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from io import StringIO
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import Epoch


def ingest_epochs(session: Session, rows: list[dict]) -> int:
    if not rows:
        return 0
    now = datetime.now(timezone.utc)
    for row in rows:
        row.setdefault("ingest_ts", now)
    try:
        return _copy_insert_epochs(session, rows)
    except Exception:
        return _insert_epochs(session, rows)


def _copy_insert_epochs(session: Session, rows: list[dict]) -> int:
    connection = session.connection()
    raw_conn = cast(Any, connection.connection)
    session.execute(
        text(
            "CREATE TEMP TABLE IF NOT EXISTS temp_epochs (LIKE epochs INCLUDING DEFAULTS) ON COMMIT DROP"
        )
    )
    buffer = StringIO()
    writer = csv.writer(buffer)
    for row in rows:
        writer.writerow(
            [
                row["tenant_id"],
                row["recording_id"],
                row["epoch_index"],
                row["epoch_start_ts"].isoformat(),
                row["feature_schema_version"],
                json.dumps(row["features_payload"]),
                row["ingest_ts"].isoformat(),
            ]
        )
    buffer.seek(0)
    with raw_conn.cursor() as cursor:
        with cursor.copy(
            "COPY temp_epochs (tenant_id, recording_id, epoch_index, epoch_start_ts, feature_schema_version, features_payload, ingest_ts) FROM STDIN WITH (FORMAT CSV)"
        ) as copy:
            copy.write(buffer.getvalue())
    result = cast(
        Any,
        session.execute(
            text(
                "INSERT INTO epochs (tenant_id, recording_id, epoch_index, epoch_start_ts, feature_schema_version, features_payload, ingest_ts)"
                " SELECT tenant_id, recording_id, epoch_index, epoch_start_ts, feature_schema_version, features_payload, ingest_ts"
                " FROM temp_epochs"
                " ON CONFLICT DO NOTHING"
            )
        ),
    )
    return int(result.rowcount or 0)


def _insert_epochs(session: Session, rows: list[dict]) -> int:
    stmt = insert(Epoch).values(rows)
    stmt = stmt.on_conflict_do_nothing(index_elements=[Epoch.recording_id, Epoch.epoch_start_ts])
    result = session.execute(stmt.returning(Epoch.recording_id))
    return len(result.fetchall())
