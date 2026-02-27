from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from extractor_hardened.errors import ExtractionError
from extractor_hardened.manifest import REQUIRED_MANIFEST_FIELDS


def validate_artifact_dir(path: str | Path) -> None:
    artifact_dir = Path(path)
    required = [
        artifact_dir / "features.npy",
        artifact_dir / "labels.npy",
        artifact_dir / "valid_mask.npy",
        artifact_dir / "timestamps.npy",
        artifact_dir / "manifest.json",
        artifact_dir / "night_summary.json",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise ExtractionError(
            "E_CONTRACT_VIOLATION", "Missing artifact files", {"missing": missing}
        )

    features = np.load(artifact_dir / "features.npy")
    labels = np.load(artifact_dir / "labels.npy")
    valid_mask = np.load(artifact_dir / "valid_mask.npy")
    timestamps = np.load(artifact_dir / "timestamps.npy")
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))

    if features.dtype != np.float32:
        raise ExtractionError("E_CONTRACT_VIOLATION", "features.npy dtype must be float32")
    if labels.dtype != np.int8:
        raise ExtractionError("E_CONTRACT_VIOLATION", "labels.npy dtype must be int8")
    if valid_mask.dtype != np.bool_:
        raise ExtractionError("E_CONTRACT_VIOLATION", "valid_mask.npy dtype must be bool")
    if timestamps.dtype != np.int64:
        raise ExtractionError("E_CONTRACT_VIOLATION", "timestamps.npy dtype must be int64")
    if features.shape[0] != labels.shape[0] or labels.shape[0] != valid_mask.shape[0]:
        raise ExtractionError("E_CONTRACT_VIOLATION", "Tensor first dimension mismatch")
    if timestamps.shape[0] != labels.shape[0]:
        raise ExtractionError("E_CONTRACT_VIOLATION", "timestamps length mismatch")

    domain = {-1, 0, 1, 2, 3, 4}
    if not set(labels.tolist()).issubset(domain):
        raise ExtractionError("E_CONTRACT_VIOLATION", "labels outside domain")

    missing_manifest_fields = [
        key for key in REQUIRED_MANIFEST_FIELDS if key not in manifest or manifest[key] is None
    ]
    if missing_manifest_fields:
        raise ExtractionError(
            "E_CONTRACT_VIOLATION",
            "manifest missing required fields",
            {"missing_fields": missing_manifest_fields},
        )

    feature_list = manifest.get("feature_list", [])
    if any(not bool(item.get("causal", False)) for item in feature_list if isinstance(item, dict)):
        raise ExtractionError(
            "E_CONTRACT_VIOLATION", "feature manifest contains non-causal feature"
        )

    alignment = manifest.get("alignment_decisions", {})
    if isinstance(alignment, dict):
        requires_invalid_mask = any(
            isinstance(decision, dict)
            and decision.get("status") == "reconciled"
            and str(decision.get("reason")) in {"short", "pad"}
            for decision in alignment.values()
        )
        if requires_invalid_mask and bool(np.all(valid_mask)):
            raise ExtractionError(
                "E_CONTRACT_VIOLATION",
                "alignment requires invalid epochs but valid mask has none",
            )

    qc_summary = manifest.get("qc_summary", {})
    if isinstance(qc_summary, dict) and qc_summary.get("failed_required_channels"):
        raise ExtractionError(
            "E_CONTRACT_VIOLATION", "manifest includes failed required QC channels"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate extractor artifact directory")
    parser.add_argument("artifact_dir")
    args = parser.parse_args()
    validate_artifact_dir(args.artifact_dir)


if __name__ == "__main__":
    main()
