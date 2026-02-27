from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from extractor_hardened.contracts import Contracts, hash_file, hash_payload
from extractor_hardened.errors import ExtractionError


REQUIRED_MANIFEST_FIELDS = [
    "raw_input_sha256",
    "extractor_version",
    "config_hash",
    "schema_hash",
    "selected_channels",
    "fs_per_channel",
    "staging_source",
    "staging_hash",
    "alignment_decisions",
    "qc_summary",
    "class_distribution",
    "feature_list",
    "record_id",
    "night_id",
]


def build_manifest(
    *,
    input_path: Path,
    record_id: str,
    night_id: str,
    config_payload: dict[str, Any],
    selected_channels: dict[str, str],
    fs_map: dict[str, float],
    alignment_decisions: dict[str, Any],
    qc_summary: dict[str, Any],
    labels: np.ndarray,
    feature_manifest: dict[str, Any],
    contracts: Contracts,
) -> dict[str, Any]:
    if not input_path.exists():
        raise ExtractionError(
            code="E_STAGING_MISSING",
            message="Input staging file is missing",
            details={"input_path": str(input_path)},
        )

    class_distribution = {str(stage): int(np.sum(labels == stage)) for stage in (-1, 0, 1, 2, 3, 4)}
    git_commit = _git_commit()
    manifest = {
        "record_id": record_id,
        "night_id": night_id,
        "raw_input_sha256": hash_file(input_path),
        "extractor_version": "0.2.0",
        "git_commit": git_commit,
        "config_hash": hash_payload(config_payload),
        "schema_hash": contracts.schema_hash,
        "selected_channels": selected_channels,
        "fs_per_channel": fs_map,
        "staging_source": str(input_path),
        "staging_hash": hash_file(input_path),
        "alignment_decisions": alignment_decisions,
        "qc_summary": qc_summary,
        "class_distribution": class_distribution,
        "feature_list": feature_manifest.get("features", []),
    }
    _validate_manifest(manifest)
    return manifest


def _git_commit() -> str | None:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None
    return output or None


def _validate_manifest(manifest: dict[str, Any]) -> None:
    missing = [
        key for key in REQUIRED_MANIFEST_FIELDS if key not in manifest or manifest[key] is None
    ]
    if missing:
        raise ExtractionError(
            code="E_CONTRACT_VIOLATION",
            message="Manifest is missing required fields",
            details={"missing_fields": missing},
        )


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
