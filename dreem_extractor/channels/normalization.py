from __future__ import annotations

import numpy as np


def normalize_signal(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(float, copy=False)
    if np.any(~np.isfinite(data)):
        data = np.nan_to_num(data, nan=0.0)
    return data


def pad_or_trim(data: np.ndarray, target_len: int) -> np.ndarray:
    if data.size == target_len:
        return data
    if data.size > target_len:
        return data[:target_len]
    pad_width = target_len - data.size
    return np.pad(data, (0, pad_width), mode="constant", constant_values=0.0)
