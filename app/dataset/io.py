from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np


def save_npz(
    output_dir: Path,
    *,
    X: np.ndarray,
    y: np.ndarray,
    label_map: list[str],
    window_end_ts: np.ndarray,
    recording_ids: np.ndarray,
    splits: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "dataset.npz",
        X=X,
        y=y,
        window_end_ts=window_end_ts,
        recording_ids=recording_ids,
        label_map=np.asarray(label_map),
        split_train=cast(np.ndarray, splits.get("train")),
        split_val=cast(np.ndarray, splits.get("val")),
        split_test=cast(np.ndarray, splits.get("test")),
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def export_parquet(
    output_dir: Path,
    *,
    X: np.ndarray,
    y: np.ndarray,
    label_map: list[str],
    window_end_ts: np.ndarray,
    recording_ids: np.ndarray,
    splits: dict[str, np.ndarray],
    metadata: dict[str, Any],
    feature_names: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for parquet export") from exc

    flat = X.reshape(X.shape[0], -1)
    columns = _flatten_feature_names(feature_names, X.shape[1])
    df = pd.DataFrame(flat, columns=columns)
    df["label"] = y
    df["window_end_ts"] = window_end_ts
    df["recording_id"] = recording_ids
    df.to_parquet(output_dir / "dataset.parquet", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    _write_splits(output_dir, splits)
    (output_dir / "label_map.json").write_text(json.dumps(label_map, indent=2))


def export_hdf5(
    output_dir: Path,
    *,
    X: np.ndarray,
    y: np.ndarray,
    label_map: list[str],
    window_end_ts: np.ndarray,
    recording_ids: np.ndarray,
    splits: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for HDF5 export") from exc
    path = output_dir / "dataset.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=X)
        handle.create_dataset("y", data=y.astype("S"))
        handle.create_dataset("window_end_ts", data=window_end_ts.astype("S"))
        handle.create_dataset("recording_ids", data=recording_ids.astype("S"))
        handle.create_dataset("label_map", data=np.asarray(label_map).astype("S"))
        for name, indices in splits.items():
            handle.create_dataset(f"split_{name}", data=indices)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _write_splits(output_dir: Path, splits: dict[str, np.ndarray]) -> None:
    for name, indices in splits.items():
        np.save(output_dir / f"split_{name}.npy", indices)


def _flatten_feature_names(feature_names: list[str], window_size: int) -> list[str]:
    columns: list[str] = []
    for step in range(window_size):
        for name in feature_names:
            columns.append(f"t{step}_{name}")
    return columns
