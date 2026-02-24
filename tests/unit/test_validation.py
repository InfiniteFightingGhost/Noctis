from __future__ import annotations

import numpy as np
import pytest

from app.ml.validation import ensure_finite, prepare_batch


def test_ensure_finite_raises_on_nan() -> None:
    data = np.array([0.0, np.nan], dtype=np.float32)
    with pytest.raises(ValueError):
        ensure_finite("nan", data)


def test_prepare_batch_mean_strategy() -> None:
    windows = [np.zeros((3, 2), dtype=np.float32), np.ones((3, 2), dtype=np.float32)]
    batch = prepare_batch(
        windows,
        feature_strategy="mean",
        expected_input_dim=2,
        feature_dim=2,
        window_size=3,
    )
    assert batch.shape == (2, 2)


def test_prepare_batch_flatten_strategy() -> None:
    windows = [np.zeros((2, 3), dtype=np.float32)]
    batch = prepare_batch(
        windows,
        feature_strategy="flatten",
        expected_input_dim=6,
        feature_dim=3,
        window_size=2,
    )
    assert batch.shape == (1, 6)


def test_prepare_batch_handles_outlier_values() -> None:
    windows = [np.full((4, 2), 1e6, dtype=np.float32)]
    batch = prepare_batch(
        windows,
        feature_strategy="mean",
        expected_input_dim=2,
        feature_dim=2,
        window_size=4,
    )
    assert batch.shape == (1, 2)


def test_prepare_batch_sequence_strategy() -> None:
    windows = [np.zeros((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32)]
    batch = prepare_batch(
        windows,
        feature_strategy="sequence",
        expected_input_dim=3,
        feature_dim=3,
        window_size=2,
    )
    assert batch.shape == (2, 2, 3)
