from __future__ import annotations

from pathlib import Path

import numpy as np

from extractor.config import FEATURE_KEYS


def build_feature_matrix(features: dict[str, np.ndarray]) -> np.ndarray:
    arrays = [features[key] for key in FEATURE_KEYS]
    return np.stack(arrays, axis=1).astype(np.float32)


def build_windows(
    features: dict[str, np.ndarray],
    stages: np.ndarray,
    stage_known: np.ndarray,
    window_len: int,
    stride: int,
    label_mode: str,
    drop_unknown: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    matrix = build_feature_matrix(features)
    num_epochs = matrix.shape[0]
    if num_epochs < window_len:
        empty_y = (
            np.empty((0, window_len), dtype=np.uint8)
            if label_mode == "sequence"
            else np.empty((0,), dtype=np.uint8)
        )
        return (
            np.empty((0, window_len, matrix.shape[1]), dtype=np.float32),
            empty_y,
            None,
        )
    windows = []
    labels = []
    masks = []
    for start in range(0, num_epochs - window_len + 1, stride):
        end = start + window_len
        window = matrix[start:end]
        if label_mode == "sequence":
            label = stages[start:end]
            known = stage_known[start:end]
        else:
            center = start + window_len // 2
            label = stages[center]
            known = stage_known[center]

        if drop_unknown:
            if label_mode == "sequence" and not np.all(known):
                continue
            if label_mode != "sequence" and not bool(known):
                continue
        if not drop_unknown:
            if label_mode == "sequence":
                label = np.where(known, label, 255)
            else:
                label = 255 if not bool(known) else label
            masks.append(known)
        windows.append(window)
        labels.append(label)

    if not windows:
        empty_y = (
            np.empty((0, window_len), dtype=np.uint8)
            if label_mode == "sequence"
            else np.empty((0,), dtype=np.uint8)
        )
        return (
            np.empty((0, window_len, matrix.shape[1]), dtype=np.float32),
            empty_y,
            None,
        )

    X = np.stack(windows, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.uint8)
    mask = None
    if not drop_unknown:
        mask = np.asarray(masks)
    return X, y, mask


def write_npz(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray | None,
    path: str | Path,
) -> None:
    path = Path(path)
    if mask is None:
        np.savez(path, X=X, y=y)
    else:
        np.savez(path, X=X, y=y, mask=mask)
