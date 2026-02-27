from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import uuid

import numpy as np
import pytest

from app.training.config import training_config_from_payload
from app.training import trainer


@dataclass(frozen=True)
class _FakeFeature:
    name: str
    dtype: str = "float32"
    allowed_range: None = None
    description: None = None
    introduced_in_version: str = "v1"
    deprecated_in_version: None = None
    position: int = 0


@dataclass(frozen=True)
class _FakeFeatureSchema:
    version: str = "v1"
    size: int = 3
    schema_hash: str = "schema-hash"


@dataclass(frozen=True)
class _FakeSchemaRecord:
    id: uuid.UUID
    version: str
    hash: str
    features: list[_FakeFeature]


def _write_dataset(dataset_dir: Path, *, X: np.ndarray, y: np.ndarray) -> None:
    np.savez_compressed(
        dataset_dir / "dataset.npz",
        X=X,
        y=y,
        label_map=np.asarray(["N1", "N2", "N3", "REM", "WAKE"]),
        recording_ids=np.asarray([f"r{i}" for i in range(X.shape[0])]),
        dataset_ids=np.asarray(["DODH", "CAP", "ISRUC", "DODH", "CAP", "ISRUC"][: X.shape[0]]),
        split_train=np.asarray([0, 1, 2, 3, 4]),
        split_val=np.asarray([], dtype=int),
        split_test=np.asarray([5]),
    )
    (dataset_dir / "metadata.json").write_text(
        json.dumps(
            {
                "window_size": int(X.shape[1]),
                "epoch_seconds": 30,
                "split_policy": {
                    "split_strategy": "recording",
                    "seed": 42,
                    "grouping_key": "recording_id",
                    "time_aware": False,
                },
                "label_source_policy": "ground_truth_only",
                "feature_schema_version": "v1",
                "feature_schema_hash": "schema-hash",
            },
            indent=2,
        )
    )


def _patch_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_schema = _FakeFeatureSchema()
    fake_record = _FakeSchemaRecord(
        id=uuid.uuid4(),
        version="v1",
        hash="schema-hash",
        features=[
            _FakeFeature(name="f0", position=0),
            _FakeFeature(name="f1", position=1),
            _FakeFeature(name="f2", position=2),
        ],
    )
    monkeypatch.setattr(trainer, "load_feature_schema", lambda path: fake_schema)
    monkeypatch.setattr(
        trainer, "_resolve_feature_schema_record", lambda *args, **kwargs: fake_record
    )
    monkeypatch.setattr(trainer, "_verify_snapshot_integrity", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "_resolve_snapshot_id", lambda *args, **kwargs: str(uuid.uuid4()))


def test_train_model_dispatches_gradient_boosting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    X = np.random.default_rng(1).normal(size=(6, 4, 3)).astype(np.float32)
    y = np.asarray(["N1", "N2", "N3", "REM", "WAKE", "N2"])
    _write_dataset(dataset_dir, X=X, y=y)
    _patch_dependencies(monkeypatch)

    feature_schema_path = dataset_dir / "feature_schema.json"
    feature_schema_path.write_text("{}")
    cfg = training_config_from_payload(
        {
            "dataset_dir": str(dataset_dir),
            "output_root": str(tmp_path / "models"),
            "feature_schema_path": str(feature_schema_path),
            "model_type": "gradient_boosting",
            "feature_strategy": "mean",
            "split_strategy": "recording",
            "split_seed": 42,
            "split_grouping_key": "recording_id",
            "split_time_aware": False,
        }
    )

    result = trainer.train_model(config=cfg, version="0.0.1-test")
    assert (result.artifact_dir / "model.bin").exists()
    assert (result.artifact_dir / "split_manifest.json").exists()


def test_train_model_dispatches_cnn_bilstm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    X = np.random.default_rng(2).normal(size=(6, 4, 3)).astype(np.float32)
    y = np.asarray(["N1", "N2", "N3", "REM", "WAKE", "N2"])
    _write_dataset(dataset_dir, X=X, y=y)
    _patch_dependencies(monkeypatch)

    feature_schema_path = dataset_dir / "feature_schema.json"
    feature_schema_path.write_text("{}")
    cfg = training_config_from_payload(
        {
            "dataset_dir": str(dataset_dir),
            "output_root": str(tmp_path / "models"),
            "feature_schema_path": str(feature_schema_path),
            "model_type": "cnn_bilstm",
            "feature_strategy": "sequence",
            "split_strategy": "recording",
            "split_seed": 42,
            "split_grouping_key": "recording_id",
            "split_time_aware": False,
            "training": {"max_epochs": 1, "batch_size": 4},
            "model": {"conv_channels": [8, 16], "head_hidden_dims": [16]},
        }
    )

    result = trainer.train_model(config=cfg, version="0.0.2-test")
    assert (result.artifact_dir / "model.pt").exists()
    assert (result.artifact_dir / "split_manifest.json").exists()
