from __future__ import annotations

from datetime import datetime, timedelta, timezone
import uuid

import numpy as np
import pytest

from app.services.windowing import WindowedEpoch, build_windows


def test_build_windows_contiguous() -> None:
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    schema_id = uuid.uuid4()
    epochs = []
    for i in range(21):
        epochs.append(
            WindowedEpoch(
                epoch_index=i,
                epoch_start_ts=start + timedelta(seconds=30 * i),
                features=np.ones(10, dtype=np.float32) * i,
                feature_schema_id=schema_id,
            )
        )
    windows = build_windows(epochs, window_size=21, allow_padding=False)
    assert len(windows) == 1
    assert windows[0].tensor.shape == (21, 10)


def test_build_windows_rejects_cross_schema() -> None:
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    epochs = [
        WindowedEpoch(
            epoch_index=0,
            epoch_start_ts=start,
            features=np.ones(2, dtype=np.float32),
            feature_schema_id=uuid.uuid4(),
        ),
        WindowedEpoch(
            epoch_index=1,
            epoch_start_ts=start + timedelta(seconds=30),
            features=np.ones(2, dtype=np.float32),
            feature_schema_id=uuid.uuid4(),
        ),
    ]
    with pytest.raises(ValueError):
        build_windows(epochs, window_size=2, allow_padding=False)


def test_build_windows_padding() -> None:
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    schema_id = uuid.uuid4()
    epochs = [
        WindowedEpoch(
            epoch_index=0,
            epoch_start_ts=start,
            features=np.ones(2, dtype=np.float32),
            feature_schema_id=schema_id,
        ),
        WindowedEpoch(
            epoch_index=2,
            epoch_start_ts=start + timedelta(seconds=60),
            features=np.ones(2, dtype=np.float32) * 2,
            feature_schema_id=schema_id,
        ),
    ]
    windows = build_windows(epochs, window_size=3, allow_padding=True, epoch_seconds=30)
    assert len(windows) == 2
    assert all(window.tensor.shape == (3, 2) for window in windows)
    assert any(window.padded for window in windows)
    assert any(window.end_ts == start + timedelta(seconds=90) for window in windows)


def test_build_windows_allows_leading_padding() -> None:
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    schema_id = uuid.uuid4()
    epochs = [
        WindowedEpoch(
            epoch_index=0,
            epoch_start_ts=start,
            features=np.ones(2, dtype=np.float32),
            feature_schema_id=schema_id,
        ),
        WindowedEpoch(
            epoch_index=1,
            epoch_start_ts=start + timedelta(seconds=30),
            features=np.ones(2, dtype=np.float32),
            feature_schema_id=schema_id,
        ),
    ]
    windows = build_windows(epochs, window_size=3, allow_padding=True, epoch_seconds=30)
    assert len(windows) == 2
    assert windows[0].padded is True
    assert windows[0].tensor.shape == (3, 2)


def test_build_windows_gap_raises_without_padding() -> None:
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    schema_id = uuid.uuid4()
    epochs = [
        WindowedEpoch(
            epoch_index=0,
            epoch_start_ts=start,
            features=np.ones(2, dtype=np.float32),
            feature_schema_id=schema_id,
        ),
        WindowedEpoch(
            epoch_index=2,
            epoch_start_ts=start + timedelta(seconds=60),
            features=np.ones(2, dtype=np.float32),
            feature_schema_id=schema_id,
        ),
    ]
    with pytest.raises(ValueError):
        build_windows(epochs, window_size=3, allow_padding=False, epoch_seconds=30)


def test_build_windows_short_sequence_returns_empty() -> None:
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    schema_id = uuid.uuid4()
    epochs = [
        WindowedEpoch(
            epoch_index=0,
            epoch_start_ts=start,
            features=np.ones(2, dtype=np.float32),
            feature_schema_id=schema_id,
        )
    ]
    windows = build_windows(epochs, window_size=3, allow_padding=False, epoch_seconds=30)
    assert windows == []
