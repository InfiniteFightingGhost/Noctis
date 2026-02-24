from __future__ import annotations

import json

import numpy as np
import pytest

from extractor_hardened.errors import ExtractionError
from extractor_hardened.validator import validate_artifact_dir


def _write_artifacts(tmp_path, manifest: dict[str, object]) -> None:
    np.save(tmp_path / "features.npy", np.ones((3, 2), dtype=np.float32))
    np.save(tmp_path / "labels.npy", np.array([0, 1, 2], dtype=np.int8))
    np.save(tmp_path / "valid_mask.npy", np.array([True, True, True], dtype=np.bool_))
    np.save(tmp_path / "timestamps.npy", np.array([0, 30, 60], dtype=np.int64))
    (tmp_path / "night_summary.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def _base_manifest() -> dict[str, object]:
    return {
        "raw_input_sha256": "abc",
        "extractor_version": "0.2.0",
        "config_hash": "cfg",
        "schema_hash": "schema",
        "selected_channels": {"ecg": "ECG"},
        "fs_per_channel": {"ecg": 50.0},
        "staging_source": "/tmp/input.h5",
        "staging_hash": "stage",
        "alignment_decisions": {"ecg": {"status": "exact"}},
        "qc_summary": {"failed_required_channels": []},
        "class_distribution": {"-1": 0, "0": 1, "1": 1, "2": 1, "3": 0, "4": 0},
        "feature_list": [{"name": "hr_mean", "causal": True}],
        "record_id": "rec-1",
        "night_id": "night-1",
    }


@pytest.mark.parametrize(
    "missing_key",
    [
        "record_id",
        "night_id",
        "selected_channels",
        "fs_per_channel",
        "staging_source",
        "staging_hash",
        "class_distribution",
        "extractor_version",
    ],
)
def test_validator_rejects_missing_manifest_contract_keys(tmp_path, missing_key: str) -> None:
    manifest = _base_manifest()
    manifest.pop(missing_key)
    _write_artifacts(tmp_path, manifest)

    with pytest.raises(ExtractionError) as exc:
        validate_artifact_dir(tmp_path)
    assert exc.value.code == "E_CONTRACT_VIOLATION"


def test_validator_rejects_none_for_required_manifest_key(tmp_path) -> None:
    manifest = _base_manifest()
    manifest["staging_hash"] = None
    _write_artifacts(tmp_path, manifest)

    with pytest.raises(ExtractionError) as exc:
        validate_artifact_dir(tmp_path)
    assert exc.value.code == "E_CONTRACT_VIOLATION"


def test_validator_accepts_full_required_manifest_contract(tmp_path) -> None:
    _write_artifacts(tmp_path, _base_manifest())
    validate_artifact_dir(tmp_path)
