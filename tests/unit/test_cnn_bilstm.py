from __future__ import annotations

import json

import numpy as np
import pytest

from app.training.cnn_bilstm import _domain_transfer_combos, train_cnn_bilstm
from app.training.config import training_config_from_payload


def _build_config(*, enable_transfer: bool):
    return training_config_from_payload(
        {
            "dataset_dir": "out/dataset",
            "output_root": "models",
            "model_type": "cnn_bilstm",
            "feature_strategy": "sequence",
            "model": {
                "use_dataset_conditioning": True,
                "conditioning_mode": "onehot",
                "conv_channels": [8, 16],
                "conv_kernel_size": 3,
                "lstm_hidden_size": 16,
                "lstm_layers": 1,
                "head_hidden_dims": [16],
            },
            "training": {
                "batch_size": 4,
                "max_epochs": 1,
                "early_stopping_patience": 1,
                "enable_domain_transfer_tests": enable_transfer,
            },
        }
    )


def test_domain_transfer_combo_ids_are_fixed() -> None:
    combos = _domain_transfer_combos()
    assert [combo["id"] for combo in combos] == [
        "train_DODH_ISRUC_test_CAP",
        "train_CAP_DODH_test_ISRUC",
        "train_CAP_ISRUC_test_DODH",
    ]


def test_train_cnn_bilstm_writes_required_artifacts(tmp_path) -> None:
    pytest.importorskip("torch")
    cfg = _build_config(enable_transfer=True)
    X = np.random.default_rng(42).normal(size=(12, 5, 3)).astype(np.float32)
    y = np.asarray(["WAKE", "N1", "N2", "N3", "REM", "N2", "WAKE", "N1", "N2", "N3", "REM", "N2"])
    dataset_ids = np.asarray(
        [
            "DODH",
            "CAP",
            "ISRUC",
            "DODH",
            "CAP",
            "ISRUC",
            "DODH",
            "CAP",
            "ISRUC",
            "DODH",
            "CAP",
            "ISRUC",
        ]
    )
    splits = {
        "train": np.asarray([0, 1, 2, 3, 4, 5, 6, 7]),
        "val": np.asarray([8, 9]),
        "test": np.asarray([10, 11]),
    }
    label_map = ["N1", "N2", "N3", "REM", "WAKE"]

    train_cnn_bilstm(
        config=cfg,
        artifact_dir=tmp_path,
        X=X,
        y=y,
        label_map=label_map,
        dataset_ids=dataset_ids,
        splits=splits,
        evaluation_split_name="test",
    )

    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "scaler.json").exists()
    assert (tmp_path / "class_distribution_report.json").exists()
    assert (tmp_path / "per_dataset_metrics.json").exists()
    assert (tmp_path / "instability_flags.json").exists()
    assert (tmp_path / "training_history.jsonl").exists()
    assert (tmp_path / "domain_transfer_report.json").exists()

    per_dataset_metrics = json.loads((tmp_path / "per_dataset_metrics.json").read_text())
    assert "aggregate" in per_dataset_metrics
