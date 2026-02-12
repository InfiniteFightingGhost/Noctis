from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from app.dataset.builder import (
    _build_windows,
    _select_label,
    _stratified_split_indices,
)
from app.dataset.config import DatasetBuildConfig
from app.services.windowing import WindowedEpoch


def test_label_alignment_ground_truth() -> None:
    label = _select_label(
        {"ground_truth_stage": "N2", "predicted_stage": "W"},
        "ground_truth_only",
    )
    assert label == "N2"


def test_label_alignment_predicted_fallback() -> None:
    label = _select_label(
        {"ground_truth_stage": None, "predicted_stage": "REM"},
        "ground_truth_or_predicted",
    )
    assert label == "REM"


def test_stratified_split_deterministic() -> None:
    y = np.array(["W", "W", "N2", "N2", "REM", "REM"])
    first = _stratified_split_indices(
        y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=123
    )
    second = _stratified_split_indices(
        y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=123
    )
    assert np.array_equal(first["train"], second["train"])
    assert np.array_equal(first["val"], second["val"])
    assert np.array_equal(first["test"], second["test"])


def test_window_build_deterministic_order() -> None:
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    epochs_a = [
        WindowedEpoch(
            i, start + timedelta(seconds=30 * i), np.ones(2, dtype=np.float32)
        )
        for i in range(21)
    ]
    epochs_b = [
        WindowedEpoch(
            i, start + timedelta(seconds=30 * i), np.ones(2, dtype=np.float32) * 2
        )
        for i in range(21)
    ]
    config = DatasetBuildConfig(
        output_dir=Path("."),
        feature_schema_path=Path("schema.json"),
        window_size=21,
    )
    windows, meta = _build_windows({"b": epochs_b, "a": epochs_a}, config)
    assert len(windows) == 2
    assert meta[0]["recording_id"] == "a"
    assert meta[1]["recording_id"] == "b"
