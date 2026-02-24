from __future__ import annotations

import json

import numpy as np
import pytest

from app.ml.model import load_model
from app.training.cnn_bilstm import train_cnn_bilstm
from app.training.config import training_config_from_payload


def test_load_model_supports_torch_sequence_adapter(tmp_path) -> None:
    pytest.importorskip("torch")
    cfg = training_config_from_payload(
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
            },
        }
    )

    X = np.random.default_rng(7).normal(size=(9, 4, 3)).astype(np.float32)
    y = np.asarray(["WAKE", "N1", "N2", "N3", "REM", "WAKE", "N1", "N2", "N3"])
    dataset_ids = np.asarray(
        ["DODH", "CAP", "ISRUC", "DODH", "CAP", "ISRUC", "DODH", "CAP", "ISRUC"]
    )
    splits = {
        "train": np.asarray([0, 1, 2, 3, 4, 5]),
        "val": np.asarray([6]),
        "test": np.asarray([7, 8]),
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

    training_config_payload = {
        "model": {
            "use_dataset_conditioning": True,
            "conditioning_mode": "onehot",
            "conditioning_embed_dim": 8,
            "conv_channels": [8, 16],
            "conv_kernel_size": 3,
            "conv_dropout": 0.1,
            "lstm_hidden_size": 16,
            "lstm_layers": 1,
            "lstm_dropout": 0.1,
            "head_hidden_dims": [16],
            "head_dropout": 0.2,
        }
    }
    (tmp_path / "training_config.json").write_text(json.dumps(training_config_payload, indent=2))
    (tmp_path / "label_map.json").write_text(json.dumps(label_map, indent=2))
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "expected_input_dim": 3,
                "dataset_id_map": {"DODH": 0, "CAP": 1, "ISRUC": 2, "UNKNOWN": 3},
            },
            indent=2,
        )
    )

    bundle = load_model(tmp_path)
    probs = bundle.model.predict_proba(
        np.random.default_rng(8).normal(size=(2, 4, 3)).astype(np.float32),
        dataset_ids=np.asarray(["DODH", "CAP"]),
    )
    assert probs.shape == (2, 5)
