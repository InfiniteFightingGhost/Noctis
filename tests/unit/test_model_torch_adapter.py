from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest

from app.ml.model import load_model
from app.training.cnn_bilstm import CnnBiLstmNetwork, train_cnn_bilstm
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
                "lstm_hidden_size": 128,
                "lstm_layers": 2,
                "head_hidden_dims": [256],
            },
            "training": {
                "batch_size": 4,
                "max_epochs": 40,
                "early_stopping_patience": 6,
            },
        }
    )

    X = np.random.default_rng(7).normal(size=(9, 4, 15)).astype(np.float32)
    y = np.asarray(["W", "Light", "Deep", "REM", "W", "Light", "Deep", "REM", "W"])
    dataset_ids = np.asarray(
        ["DODH", "CAP", "ISRUC", "SLEEP-EDF", "CAP", "ISRUC", "DODH", "CAP", "ISRUC"]
    )
    feature_names = [
        "in_bed_pct",
        "hr_mean",
        "hr_std",
        "dhr",
        "rr_mean",
        "rr_std",
        "drr",
        "large_move_pct",
        "minor_move_pct",
        "turnovers_delta",
        "apnea_delta",
        "flags",
        "vib_move_pct",
        "vib_resp_q",
        "agree_flags",
    ]
    recording_ids = np.asarray(["r1", "r1", "r1", "r2", "r2", "r2", "r3", "r3", "r3"])
    splits: dict[str, np.ndarray | None] = {
        "train": np.asarray([0, 1, 2, 3, 4, 5]),
        "val": np.asarray([6]),
        "test": np.asarray([7, 8]),
    }
    label_map = ["W", "Light", "Deep", "REM"]

    train_cnn_bilstm(
        config=cfg,
        artifact_dir=tmp_path,
        X=X,
        y=y,
        label_map=label_map,
        feature_names=feature_names,
        dataset_ids=dataset_ids,
        recording_ids=recording_ids,
        splits=cast(dict[str, np.ndarray | None], splits),
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
            "lstm_hidden_size": 128,
            "lstm_layers": 2,
            "lstm_dropout": 0.1,
            "head_hidden_dims": [256],
            "head_dropout": 0.2,
        }
    }
    (tmp_path / "training_config.json").write_text(json.dumps(training_config_payload, indent=2))
    (tmp_path / "label_map.json").write_text(json.dumps(label_map, indent=2))
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "expected_input_dim": 26,
                "base_input_dim": 15,
                "dataset_id_map": {"DODH": 0, "CAP": 1, "ISRUC": 2, "SLEEP-EDF": 3, "UNKNOWN": 4},
            },
            indent=2,
        )
    )

    bundle = load_model(tmp_path)
    penalties = bundle.model.transition_penalties(["W", "Light", "Deep", "REM"])
    assert penalties is None or penalties.shape == (4, 4)
    probs = bundle.model.predict_proba(
        np.random.default_rng(8).normal(size=(2, 4, 15)).astype(np.float32),
        dataset_ids=np.asarray(["DODH", "CAP"]),
    )
    assert probs.shape == (2, 4)


def _write_sequence_artifacts(tmp_path, *, legacy_head: bool) -> None:
    torch = pytest.importorskip("torch")
    model_cfg = {
        "use_dataset_conditioning": True,
        "conditioning_mode": "onehot",
        "conditioning_embed_dim": 8,
        "conv_channels": [8, 16],
        "conv_kernel_size": 3,
        "conv_dropout": 0.1,
        "lstm_hidden_size": 32,
        "lstm_layers": 1,
        "lstm_dropout": 0.1,
        "head_hidden_dims": [16],
        "head_dropout": 0.2,
    }
    label_map = ["W", "Light", "Deep", "REM"]
    network = CnnBiLstmNetwork(
        input_dim=15,
        num_classes=len(label_map),
        num_domains=5,
        model_cfg=model_cfg,
        torch=torch,
        nn=torch.nn,
    )
    state = network.module.state_dict()
    if legacy_head:
        legacy_state = {}
        for key, value in state.items():
            if key.startswith("aux_head."):
                continue
            if key.startswith("primary_head."):
                legacy_state[f"head.{key[len('primary_head.'):]}"] = value
            else:
                legacy_state[key] = value
        state = legacy_state
    torch.save(state, tmp_path / "model.pt")
    (tmp_path / "training_config.json").write_text(json.dumps({"model": model_cfg}, indent=2))
    (tmp_path / "label_map.json").write_text(json.dumps(label_map, indent=2))
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "feature_strategy": "sequence",
                "expected_input_dim": 15,
                "base_input_dim": 15,
                "dataset_id_map": {
                    "DODH": 0,
                    "CAP": 1,
                    "ISRUC": 2,
                    "SLEEP-EDF": 3,
                    "UNKNOWN": 4,
                },
            },
            indent=2,
        )
    )


def test_load_model_supports_legacy_head_checkpoint_keys(tmp_path) -> None:
    _write_sequence_artifacts(tmp_path, legacy_head=True)
    bundle = load_model(tmp_path)
    probs = bundle.model.predict_proba(
        np.random.default_rng(9).normal(size=(3, 4, 15)).astype(np.float32),
        dataset_ids=np.asarray(["UNKNOWN", "CAP", "DODH"]),
    )
    assert probs.shape == (3, 4)


def test_load_model_supports_new_primary_aux_checkpoint_keys(tmp_path) -> None:
    _write_sequence_artifacts(tmp_path, legacy_head=False)
    bundle = load_model(tmp_path)
    probs = bundle.model.predict_proba(
        np.random.default_rng(10).normal(size=(2, 4, 15)).astype(np.float32),
        dataset_ids=np.asarray(["ISRUC", "UNKNOWN"]),
    )
    assert probs.shape == (2, 4)
