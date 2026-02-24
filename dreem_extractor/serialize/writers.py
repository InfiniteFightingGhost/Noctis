from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from extractor_hardened.artifacts import write_record_artifacts
from extractor_hardened.contracts import load_contracts
from extractor_hardened.manifest import build_manifest
from extractor_hardened.night_summary import build_night_summary
from dreem_extractor.models import ExtractResult


def write_record_outputs(result: ExtractResult, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contracts = load_contracts()
    labels = result.hypnogram.astype(np.int8)
    valid_mask = result.valid_mask.astype(bool)
    timestamps = (
        result.timestamps.astype(np.int64)
        if result.timestamps is not None
        else np.arange(labels.shape[0], dtype=np.int64) * 30
    )

    features = result.features.astype(np.float32)
    features[features == 255] = np.nan

    manifest = build_manifest(
        input_path=Path(str(result.metadata.get("source_path", ""))),
        record_id=result.record_id,
        night_id=result.record_id,
        config_payload=dict(result.metadata.get("config_payload", {})),
        selected_channels=dict(result.metadata.get("channel_map", {})),
        fs_map=dict(result.metadata.get("fs_map", {})),
        alignment_decisions=dict(result.metadata.get("alignment_decisions", {})),
        qc_summary=dict(result.metadata.get("qc_summary", {})),
        labels=labels,
        feature_manifest=contracts.feature_manifest,
        contracts=contracts,
    )
    summary = build_night_summary(labels=labels, valid_mask=valid_mask)
    record_dir = write_record_artifacts(
        out_dir=out_dir,
        record_id=result.record_id,
        features=features,
        labels=labels,
        valid_mask=valid_mask,
        timestamps=timestamps,
        manifest=manifest,
        night_summary=summary,
    )
    return {
        "record_dir": record_dir,
        "features": record_dir / "features.npy",
        "manifest": record_dir / "manifest.json",
    }


def write_manifest(entries: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
