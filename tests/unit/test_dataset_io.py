from __future__ import annotations

import numpy as np

from app.dataset.io import save_npz


def test_save_npz_persists_dataset_ids(tmp_path) -> None:
    save_npz(
        tmp_path,
        X=np.zeros((2, 3, 4), dtype=np.float32),
        y=np.asarray(["N1", "N2"]),
        label_map=["N1", "N2"],
        window_end_ts=np.asarray(["2026-01-01T00:00:00+00:00", "2026-01-01T00:00:30+00:00"]),
        recording_ids=np.asarray(["r1", "r2"]),
        dataset_ids=np.asarray(["DODH", "CAP"]),
        splits={
            "train": np.asarray([0]),
            "val": np.asarray([], dtype=int),
            "test": np.asarray([1]),
        },
        metadata={"window_size": 3},
    )
    payload = np.load(tmp_path / "dataset.npz", allow_pickle=True)
    assert "dataset_ids" in payload
    assert payload["dataset_ids"].tolist() == ["DODH", "CAP"]


def test_save_npz_defaults_unknown_dataset_ids(tmp_path) -> None:
    save_npz(
        tmp_path,
        X=np.zeros((1, 3, 2), dtype=np.float32),
        y=np.asarray(["N2"]),
        label_map=["N2"],
        window_end_ts=np.asarray(["2026-01-01T00:00:00+00:00"]),
        recording_ids=np.asarray(["r1"]),
        splits={
            "train": np.asarray([0]),
            "val": np.asarray([], dtype=int),
            "test": np.asarray([], dtype=int),
        },
        metadata={"window_size": 3},
    )
    payload = np.load(tmp_path / "dataset.npz", allow_pickle=True)
    assert payload["dataset_ids"].tolist() == ["UNKNOWN"]
