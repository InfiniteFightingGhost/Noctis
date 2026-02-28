from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest

from app.ml.decoding import (
    transition_penalty_matrix,
    viterbi_decode_probabilities,
    viterbi_decode_probabilities_with_penalties,
)
from app.training.cnn_bilstm import train_cnn_bilstm
from app.training.config import training_config_from_payload
from app.training.mmwave import ENGINEERED_FEATURE_NAMES, engineer_mmwave_features


def _config():
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
                "lstm_hidden_size": 128,
                "lstm_layers": 2,
                "head_hidden_dims": [256],
            },
            "training": {
                "batch_size": 4,
                "max_epochs": 40,
                "early_stopping_patience": 6,
                "scheduler_patience": 2,
                "enable_binary_pretraining": True,
                "pretrain_epochs": 2,
            },
        }
    )


def test_engineered_features_are_deterministic() -> None:
    X = np.random.default_rng(42).normal(size=(2, 3, 15)).astype(np.float32)
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
    first, names, _ = engineer_mmwave_features(
        X,
        feature_names=feature_names,
        low_agreement_threshold=0.2,
    )
    second, _, _ = engineer_mmwave_features(
        X,
        feature_names=feature_names,
        low_agreement_threshold=0.2,
    )
    assert np.array_equal(first, second)
    assert names[-len(ENGINEERED_FEATURE_NAMES) :] == ENGINEERED_FEATURE_NAMES


def test_viterbi_penalizes_forbidden_jumps() -> None:
    labels = ["W", "Light", "Deep", "REM"]
    penalties = transition_penalty_matrix(labels)
    assert (
        penalties[labels.index("W"), labels.index("Deep")]
        > penalties[labels.index("W"), labels.index("Light")]
    )
    probs = np.asarray(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.6, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    decoded = viterbi_decode_probabilities(probs, labels)
    assert decoded.shape[0] == 3


def test_viterbi_accepts_explicit_transition_penalties() -> None:
    labels = ["W", "Light", "Deep", "REM"]
    probs = np.asarray(
        [
            [0.6, 0.3, 0.1, 0.0],
            [0.2, 0.5, 0.2, 0.1],
            [0.1, 0.4, 0.2, 0.3],
        ],
        dtype=np.float32,
    )
    penalties = np.ones((4, 4), dtype=np.float32)
    np.fill_diagonal(penalties, 0.0)
    decoded = viterbi_decode_probabilities_with_penalties(
        probs,
        labels,
        transition_penalties=penalties,
    )
    assert decoded.shape == (3,)


def test_train_cnn_bilstm_reports_pre_and_post_decode(tmp_path) -> None:
    pytest.importorskip("torch")
    cfg = _config()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, 5, 15)).astype(np.float32)
    y = np.asarray(
        [
            "W",
            "Light",
            "Deep",
            "REM",
            "W",
            "Light",
            "Deep",
            "REM",
            "W",
            "Light",
            "Deep",
            "REM",
            "W",
            "Light",
            "Deep",
            "REM",
        ]
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
    dataset_ids = np.asarray(["CAP", "ISRUC", "SLEEP-EDF", "CAP"] * 4)
    recording_ids = np.asarray([f"r{i//4}" for i in range(16)])
    splits: dict[str, np.ndarray | None] = {
        "train": np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        "val": np.asarray([12, 13]),
        "test": np.asarray([14, 15]),
    }

    output = train_cnn_bilstm(
        config=cfg,
        artifact_dir=tmp_path,
        X=X,
        y=y,
        label_map=["W", "Light", "Deep", "REM"],
        feature_names=feature_names,
        dataset_ids=dataset_ids,
        recording_ids=recording_ids,
        splits=cast(dict[str, np.ndarray | None], splits),
        evaluation_split_name="test",
    )

    assert "pre_decode" in output.metrics
    assert "post_decode" in output.metrics
    assert "ece" in output.metrics["post_decode"]
    assert "classification" in output.metrics
    assert "global" in output.metrics["classification"]
    assert output.metrics["training"]["binary_pretraining_enabled"] is True
    assert "head_type" in output.metrics["training"]
    assert (tmp_path / "temperature.json").exists()
    assert (tmp_path / "transition_matrix.json").exists()
    assert (tmp_path / "metrics_epoch.jsonl").exists()
    assert (tmp_path / "per_dataset_metrics.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "calibration.json").exists()
    assert (tmp_path / "night_metrics.json").exists()
    assert (tmp_path / "domain_metrics.json").exists()
    assert (tmp_path / "robustness_report.json").exists()
    payload = json.loads((tmp_path / "feature_pipeline.json").read_text())
    assert payload["eps"] == 1e-6
