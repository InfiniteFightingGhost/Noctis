from __future__ import annotations

import numpy as np


def ensure_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


def prepare_batch(
    windows: list[np.ndarray],
    *,
    feature_strategy: str,
    expected_input_dim: int,
    feature_dim: int,
    window_size: int,
) -> np.ndarray:
    if not windows:
        raise ValueError("No windows provided")
    for window in windows:
        if window.ndim != 2:
            raise ValueError("Window tensor must be 2D")
        if int(window.shape[1]) != feature_dim:
            raise ValueError("Window tensor has invalid feature dimension")
        ensure_finite("window", window)
    batch = np.stack(windows, axis=0)
    if batch.ndim != 3:
        raise ValueError("Batch tensor must be 3D")
    if feature_strategy == "mean":
        batch = batch.mean(axis=1)
    elif feature_strategy == "flatten":
        if int(batch.shape[1]) != window_size:
            raise ValueError("Window tensor has invalid time dimension")
        batch = batch.reshape(batch.shape[0], -1)
    else:
        raise ValueError("Unknown feature strategy")
    ensure_finite("batch", batch)
    if int(batch.shape[1]) != int(expected_input_dim):
        raise ValueError("Batch feature dimension mismatch")
    return batch
