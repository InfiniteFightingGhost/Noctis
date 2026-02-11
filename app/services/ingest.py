from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import Epoch


def ingest_epochs(session: Session, rows: list[dict]) -> int:
    if not rows:
        return 0
    stmt = insert(Epoch).values(rows)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=[Epoch.recording_id, Epoch.epoch_start_ts]
    )
    result = session.execute(stmt.returning(Epoch.recording_id))
    inserted = len(result.fetchall())
    session.commit()
    return inserted
