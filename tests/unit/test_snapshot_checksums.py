from __future__ import annotations

from app.reproducibility.snapshots import compute_snapshot_checksum


def test_snapshot_checksum_deterministic() -> None:
    entries = [("rec1:ts1", "N2"), ("rec2:ts2", "REM")]
    first = compute_snapshot_checksum(entries)
    second = compute_snapshot_checksum(entries)
    assert first == second


def test_snapshot_checksum_changes_with_labels() -> None:
    entries = [("rec1:ts1", "N2"), ("rec2:ts2", "REM")]
    modified = [("rec1:ts1", "N2"), ("rec2:ts2", "W")]
    assert compute_snapshot_checksum(entries) != compute_snapshot_checksum(modified)
