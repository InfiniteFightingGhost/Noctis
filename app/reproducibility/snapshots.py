from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DatasetSnapshot, DatasetSnapshotWindow


@dataclass(frozen=True)
class SnapshotChecksumResult:
    snapshot_id: uuid.UUID
    expected_checksum: str
    computed_checksum: str
    row_count: int
    computed_row_count: int

    @property
    def matches(self) -> bool:
        return (
            self.expected_checksum == self.computed_checksum
            and self.row_count == self.computed_row_count
        )


def compute_snapshot_checksum(entries: list[tuple[str, str | None]]) -> str:
    hasher = hashlib.sha256()
    for window_id, label in entries:
        hasher.update(window_id.encode("utf-8"))
        hasher.update(b"|")
        hasher.update((label or "").encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def snapshot_window_id(recording_id: uuid.UUID, window_end_ts: datetime) -> str:
    return f"{recording_id}:{window_end_ts.isoformat()}"


def verify_snapshot_checksum(
    session: Session,
    *,
    snapshot_id: uuid.UUID,
) -> SnapshotChecksumResult:
    snapshot = session.execute(
        select(DatasetSnapshot).where(DatasetSnapshot.id == snapshot_id)
    ).scalar_one_or_none()
    if snapshot is None:
        raise ValueError("Dataset snapshot not found")
    rows = session.execute(
        select(
            DatasetSnapshotWindow.recording_id,
            DatasetSnapshotWindow.window_end_ts,
            DatasetSnapshotWindow.label_value,
        )
        .where(DatasetSnapshotWindow.dataset_snapshot_id == snapshot_id)
        .order_by(DatasetSnapshotWindow.window_order)
    ).all()
    entries = [
        (snapshot_window_id(recording_id, window_end_ts), label_value)
        for recording_id, window_end_ts, label_value in rows
    ]
    computed = compute_snapshot_checksum(entries)
    return SnapshotChecksumResult(
        snapshot_id=snapshot.id,
        expected_checksum=snapshot.checksum,
        computed_checksum=computed,
        row_count=snapshot.row_count,
        computed_row_count=len(entries),
    )
