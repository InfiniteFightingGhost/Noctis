from __future__ import annotations

from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert

from app.db.models import Epoch


def test_epoch_ingest_idempotency_sql() -> None:
    stmt = insert(Epoch).values(
        recording_id="00000000-0000-0000-0000-000000000000",
        epoch_index=1,
        epoch_start_ts="2026-02-01T00:00:00Z",
        feature_schema_version="v1",
        features_payload={"features": [0.0]},
        ingest_ts="2026-02-01T00:00:00Z",
    )
    stmt = stmt.on_conflict_do_nothing(
        index_elements=[Epoch.recording_id, Epoch.epoch_start_ts]
    )
    compiled = str(stmt.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT" in compiled
