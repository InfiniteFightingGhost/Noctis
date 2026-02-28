from __future__ import annotations

import numpy as np

from app.training.config import training_config_from_payload
from app.training.trainer import _build_split_manifest, _prepare_sequence_features


def test_split_manifest_handles_missing_recording_ids() -> None:
    dataset = {}
    splits = {
        "train": np.asarray([0, 1], dtype=int),
        "val": np.asarray([2], dtype=int),
        "test": np.asarray([3], dtype=int),
    }
    manifest = _build_split_manifest(dataset, splits)
    assert manifest["recording_ids_present"] is False
    assert manifest["split_sizes"]["train"] == 2
    assert manifest["recording_overlap"] is False


def test_prepare_sequence_features_allows_non_finite_for_imputation() -> None:
    config = training_config_from_payload(
        {
            "dataset_dir": "out/dataset",
            "output_root": "models",
            "model_type": "cnn_bilstm",
            "feature_strategy": "sequence",
        }
    )
    dataset = {
        "X": np.asarray(
            [
                [
                    [0.0, np.nan, 1.0, 0.0, 1.0, 0.5, 0.1, 2.0, 1.0, 0.0, 0.0, 3.0, 1.0, 0.4, 1.0],
                    [0.0, 2.0, 1.0, 0.1, 1.2, 0.5, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0, 1.0, 0.4, 1.0],
                ]
            ],
            dtype=np.float32,
        ),
        "y": np.asarray(["N2"]),
    }
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
    X, y, label_map = _prepare_sequence_features(dataset, config, feature_names=feature_names)
    assert np.isnan(X).any()
    assert y.tolist() == ["Light"]
    assert label_map == ["W", "Light", "Deep", "REM"]
