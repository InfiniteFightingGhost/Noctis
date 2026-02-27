from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from extractor_hardened.manifest import write_manifest


def write_record_artifacts(
    *,
    out_dir: Path,
    record_id: str,
    features: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    manifest: dict[str, Any],
    night_summary: dict[str, Any],
) -> Path:
    temp_dir = out_dir / f".{record_id}.tmp"
    record_dir = out_dir / record_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    np.save(temp_dir / "features.npy", features.astype(np.float32))
    np.save(temp_dir / "labels.npy", labels.astype(np.int8))
    np.save(temp_dir / "valid_mask.npy", valid_mask.astype(bool))
    np.save(temp_dir / "timestamps.npy", timestamps.astype(np.int64))
    write_manifest(temp_dir / "manifest.json", manifest)
    write_manifest(temp_dir / "night_summary.json", night_summary)

    if record_dir.exists():
        shutil.rmtree(record_dir)
    temp_dir.rename(record_dir)
    return record_dir
