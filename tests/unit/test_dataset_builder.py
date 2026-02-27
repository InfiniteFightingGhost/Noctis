from __future__ import annotations

from datetime import datetime, timedelta, timezone
import uuid
from pathlib import Path

import numpy as np

from app.dataset.builder import (
    _balance_classes,
    _build_windows,
    _infer_dataset_id,
    _recording_split_indices,
    _select_label,
    _stratified_split_indices,
)
from app.dataset.config import DatasetBuildConfig
from app.services.windowing import WindowedEpoch


def test_label_alignment_ground_truth() -> None:
    label, source = _select_label(
        {"ground_truth_stage": "N2", "predicted_stage": "W"},
        "ground_truth_only",
    )
    assert label == "N2"
    assert source == "ground_truth"


def test_label_alignment_predicted_fallback() -> None:
    label, source = _select_label(
        {"ground_truth_stage": None, "predicted_stage": "REM"},
        "ground_truth_or_predicted",
    )
    assert label == "REM"
    assert source == "predicted"


def test_stratified_split_deterministic() -> None:
    y = np.array(["W", "W", "N2", "N2", "REM", "REM"])
    first = _stratified_split_indices(y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=123)
    second = _stratified_split_indices(
        y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=123
    )
    assert np.array_equal(first["train"], second["train"])
    assert np.array_equal(first["val"], second["val"])
    assert np.array_equal(first["test"], second["test"])


def test_window_build_deterministic_order() -> None:
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    schema_id = uuid.uuid4()
    epochs_a = [
        WindowedEpoch(
            i,
            start + timedelta(seconds=30 * i),
            np.ones(2, dtype=np.float32),
            feature_schema_id=schema_id,
        )
        for i in range(21)
    ]
    epochs_b = [
        WindowedEpoch(
            i,
            start + timedelta(seconds=30 * i),
            np.ones(2, dtype=np.float32) * 2,
            feature_schema_id=schema_id,
        )
        for i in range(21)
    ]
    config = DatasetBuildConfig(
        output_dir=Path("."),
        feature_schema_path=Path("schema.json"),
        feature_schema_version="v1",
        window_size=21,
    )
    windows, meta = _build_windows({"b": epochs_b, "a": epochs_a}, config)
    assert len(windows) == 2
    assert meta[0]["recording_id"] == "a"
    assert meta[1]["recording_id"] == "b"


def test_recording_time_split_groups_overlaps() -> None:
    recording_ids = np.array(["a", "a", "b", "b", "c", "c"])
    window_end_ts = np.array(
        [
            "2026-02-01T00:00:30+00:00",
            "2026-02-01T00:01:00+00:00",
            "2026-02-01T00:00:45+00:00",
            "2026-02-01T00:01:15+00:00",
            "2026-02-01T02:00:00+00:00",
            "2026-02-01T02:00:30+00:00",
        ]
    )
    splits = _recording_split_indices(
        recording_ids,
        window_end_ts,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=123,
        split_strategy="recording_time",
        split_time_aware=True,
        split_purge_gap=0,
        split_block_seconds=None,
    )

    def split_for(rec_id: str) -> str:
        for name, indices in splits.items():
            if any(recording_ids[idx] == rec_id for idx in indices):
                return name
        raise AssertionError("Recording not assigned")

    assert split_for("a") == split_for("b")


def test_split_purge_gap_drops_boundary_windows() -> None:
    recording_ids = np.array(["a", "a", "a", "a"])
    window_end_ts = np.array(
        [
            "2026-02-01T00:00:30+00:00",
            "2026-02-01T00:01:00+00:00",
            "2026-02-01T00:01:30+00:00",
            "2026-02-01T00:02:00+00:00",
        ]
    )
    splits = _recording_split_indices(
        recording_ids,
        window_end_ts,
        train_ratio=0.5,
        val_ratio=0.0,
        test_ratio=0.5,
        seed=123,
        split_strategy="recording",
        split_time_aware=True,
        split_purge_gap=1,
        split_block_seconds=60,
    )
    assigned = set(splits["train"]) | set(splits["test"]) | set(splits["val"])
    assert 2 not in assigned
    assert 3 not in assigned
    assert 0 in assigned
    assert 1 in assigned


def test_infer_dataset_id_from_external_id() -> None:
    assert _infer_dataset_id("subject_dodh_001") == "DODH"
    assert _infer_dataset_id("CAP-night-2") == "CAP"
    assert _infer_dataset_id("isruc_patient") == "ISRUC"
    assert _infer_dataset_id("other") == "UNKNOWN"


def test_balance_classes_preserves_dataset_id_alignment() -> None:
    X = np.arange(8, dtype=np.float32).reshape(4, 1, 2)
    y = np.asarray(["N1", "N1", "N2", "N2"])
    label_sources = np.asarray(["ground_truth", "ground_truth", "ground_truth", "ground_truth"])
    window_end_ts = np.asarray(["t1", "t2", "t3", "t4"])
    recording_ids = np.asarray(["r1", "r2", "r3", "r4"])
    dataset_ids = np.asarray(["DODH", "CAP", "ISRUC", "CAP"])

    _, y_b, _, _, recording_b, dataset_b = _balance_classes(
        X,
        y,
        label_sources,
        window_end_ts,
        recording_ids,
        dataset_ids,
        strategy="undersample",
        seed=42,
    )

    assert len(y_b) == len(recording_b) == len(dataset_b)
    aligned = dict(zip(recording_b.tolist(), dataset_b.tolist(), strict=True))
    assert aligned["r1"] == "DODH"
    assert aligned["r3"] == "ISRUC"
