from __future__ import annotations

from app.lineage.service import build_lineage_metadata


def test_build_lineage_metadata() -> None:
    payload = build_lineage_metadata(
        dataset_snapshot_id="snapshot-1",
        feature_schema_version="v1",
        feature_schema_hash="hash",
        training_seed=42,
        git_commit_hash="abc123",
        metrics_hash="metrics",
        artifact_hash="artifact",
    )
    assert payload["dataset_snapshot_id"] == "snapshot-1"
    assert payload["feature_schema_version"] == "v1"
    assert payload["training_seed"] == 42
