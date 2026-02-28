from __future__ import annotations

from typing import Iterable

import numpy as np


STAGE_MAP_STR = {
    "w": 0,
    "wake": 0,
    "n1": 1,
    "n2": 2,
    "n3": 3,
    "n4": 3,
    "r": 4,
    "rem": 4,
}


def map_hypnogram(raw: Iterable) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw)
    if values.size == 0:
        empty = np.asarray(values, dtype=np.int16)
        return empty, np.zeros_like(empty, dtype=bool)
    if values.dtype.kind in {"U", "S", "O"}:
        return map_string_stages(values)
    return map_numeric_stages(values)


def map_numeric_stages(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mapped = np.full(values.shape, -1, dtype=np.int16)
    numeric_values: list[int] = []
    for value in values.ravel():
        try:
            v = int(value)
        except (TypeError, ValueError):
            continue
        if v not in (-1, 9, 99):
            numeric_values.append(v)

    use_shifted_scheme = bool(numeric_values) and min(numeric_values) >= 1

    for idx, value in np.ndenumerate(values):
        try:
            v = int(value)
        except (TypeError, ValueError):
            continue
        if v in (-1, 9, 99):
            continue
        if use_shifted_scheme and 1 <= v <= 5:
            mapped[idx] = v - 1
        elif 0 <= v <= 4:
            mapped[idx] = v
    stage_known = mapped >= 0
    return mapped.astype(np.int16), stage_known


def map_string_stages(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mapped = np.full(values.shape, -1, dtype=np.int16)
    for idx, value in np.ndenumerate(values):
        if value is None:
            continue
        if isinstance(value, bytes):
            key = value.decode("utf-8", errors="ignore").strip().lower()
        else:
            key = str(value).strip().lower()
        key = key.replace("stage", "").replace(" ", "")
        if key in STAGE_MAP_STR:
            mapped[idx] = STAGE_MAP_STR[key]
    stage_known = mapped >= 0
    return mapped.astype(np.int16), stage_known
