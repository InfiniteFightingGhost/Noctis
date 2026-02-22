from __future__ import annotations

from typing import Any


def build_lineage_metadata(
    *,
    dataset_snapshot_id: str,
    feature_schema_version: str,
    feature_schema_hash: str,
    training_seed: int,
    git_commit_hash: str | None,
    metrics_hash: str,
    artifact_hash: str,
) -> dict[str, Any]:
    return {
        "dataset_snapshot_id": dataset_snapshot_id,
        "feature_schema_version": feature_schema_version,
        "feature_schema_hash": feature_schema_hash,
        "training_seed": training_seed,
        "git_commit_hash": git_commit_hash,
        "metrics_hash": metrics_hash,
        "artifact_hash": artifact_hash,
    }
