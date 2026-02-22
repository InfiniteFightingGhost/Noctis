from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from dreem_extractor.models import ExtractResult


def write_record_outputs(result: ExtractResult, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{result.record_id}.npz"
    meta_path = out_dir / f"{result.record_id}.json"
    qc_path = out_dir / f"{result.record_id}.qc.json"

    hypnogram = result.hypnogram.astype(np.int8)
    features = result.features.astype(np.int16)
    valid_mask = result.valid_mask.astype(bool)
    if result.timestamps is not None:
        np.savez(
            file=npz_path,
            hypnogram=hypnogram,
            features=features,
            valid_mask=valid_mask,
            timestamps=result.timestamps,
        )
    else:
        np.savez(
            file=npz_path,
            hypnogram=hypnogram,
            features=features,
            valid_mask=valid_mask,
        )
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(result.metadata, handle, indent=2)
        handle.write("\n")
    with qc_path.open("w", encoding="utf-8") as handle:
        json.dump(result.qc, handle, indent=2)
        handle.write("\n")

    return {"npz": npz_path, "metadata": meta_path, "qc": qc_path}


def write_manifest(entries: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
