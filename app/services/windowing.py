from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import uuid

import numpy as np


@dataclass(frozen=True)
class WindowedEpoch:
    epoch_index: int
    epoch_start_ts: datetime
    features: np.ndarray
    feature_schema_id: uuid.UUID


@dataclass(frozen=True)
class Window:
    start_ts: datetime
    end_ts: datetime
    tensor: np.ndarray
    feature_schema_id: uuid.UUID


def build_windows(
    epochs: list[WindowedEpoch],
    window_size: int,
    allow_padding: bool,
) -> list[Window]:
    if not epochs:
        return []
    ordered = sorted(epochs, key=lambda e: e.epoch_index)
    schema_id = ordered[0].feature_schema_id
    if any(epoch.feature_schema_id != schema_id for epoch in ordered):
        raise ValueError("Cross-schema window mixing is not allowed")
    windows: list[Window] = []
    for idx in range(len(ordered)):
        start = idx - window_size + 1
        if start < 0:
            continue
        slice_epochs = ordered[start : idx + 1]
        expected = list(
            range(
                slice_epochs[0].epoch_index, slice_epochs[0].epoch_index + window_size
            )
        )
        actual = [item.epoch_index for item in slice_epochs]
        if actual != expected:
            if not allow_padding:
                continue
            raise ValueError("Padding policy not implemented")
        tensor = np.stack([item.features for item in slice_epochs], axis=0)
        windows.append(
            Window(
                start_ts=slice_epochs[0].epoch_start_ts,
                end_ts=slice_epochs[-1].epoch_start_ts,
                tensor=tensor,
                feature_schema_id=schema_id,
            )
        )
    return windows
