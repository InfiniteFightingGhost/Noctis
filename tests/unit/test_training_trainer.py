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
        "X": np.asarray([[[np.nan, 1.0], [2.0, 3.0]]], dtype=np.float32),
        "y": np.asarray(["N2"]),
        "label_map": np.asarray(["N2"]),
    }
    X, y, label_map = _prepare_sequence_features(dataset, config)
    assert np.isnan(X).any()
    assert y.tolist() == ["N2"]
    assert label_map == ["N2"]
