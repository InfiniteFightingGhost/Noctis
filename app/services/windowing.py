from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
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
    start_index: int
    end_index: int
    padded: bool


def build_windows(
    epochs: list[WindowedEpoch],
    window_size: int,
    allow_padding: bool,
    epoch_seconds: int | None = None,
) -> list[Window]:
    if not epochs:
        return []
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    ordered = sorted(epochs, key=lambda e: e.epoch_index)
    schema_id = ordered[0].feature_schema_id
    if any(epoch.feature_schema_id != schema_id for epoch in ordered):
        raise ValueError("Cross-schema window mixing is not allowed")
    feature_dim = int(ordered[0].features.shape[0])
    index_lookup = {epoch.epoch_index: epoch for epoch in ordered}
    if epoch_seconds is None:
        epoch_seconds = 30
    windows: list[Window] = []
    for idx in range(len(ordered)):
        end_epoch = ordered[idx]
        end_index = end_epoch.epoch_index
        start_index = end_index - window_size + 1
        if start_index < 0 and not allow_padding:
            continue
        expected_indices = list(range(start_index, end_index + 1))
        padded = False
        slice_epochs: list[WindowedEpoch | None] = []
        for expected in expected_indices:
            epoch = index_lookup.get(expected)
            if epoch is None:
                if not allow_padding:
                    raise ValueError("Missing epoch index in window")
                padded = True
                slice_epochs.append(None)
            else:
                if int(epoch.features.shape[0]) != feature_dim:
                    raise ValueError("Feature dimension mismatch in window")
                slice_epochs.append(epoch)
        tensors: list[np.ndarray] = []
        for epoch in slice_epochs:
            if epoch is None:
                tensors.append(np.zeros(feature_dim, dtype=np.float32))
            else:
                tensors.append(epoch.features)
        tensor = np.stack(tensors, axis=0)
        end_ts = end_epoch.epoch_start_ts + timedelta(seconds=epoch_seconds)
        start_ts = end_ts - timedelta(seconds=epoch_seconds * window_size)
        windows.append(
            Window(
                start_ts=start_ts,
                end_ts=end_ts,
                tensor=tensor,
                feature_schema_id=schema_id,
                start_index=start_index,
                end_index=end_index,
                padded=padded,
            )
        )
    return windows
