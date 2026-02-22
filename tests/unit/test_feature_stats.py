from __future__ import annotations

from datetime import datetime, timezone
import uuid

import numpy as np

import pytest

from app.services.feature_stats import (
    compute_daily_feature_stats,
    extract_means,
    merge_feature_stats,
)
from app.services.windowing import WindowedEpoch


def test_compute_daily_feature_stats() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    schema_id = uuid.uuid4()
    epochs = [
        WindowedEpoch(
            epoch_index=0,
            epoch_start_ts=start,
            features=np.array([1.0, 2.0]),
            feature_schema_id=schema_id,
        ),
        WindowedEpoch(
            epoch_index=1,
            epoch_start_ts=start,
            features=np.array([3.0, 4.0]),
            feature_schema_id=schema_id,
        ),
    ]
    aggregates = compute_daily_feature_stats(epochs)
    assert len(aggregates) == 1
    stats = aggregates[0].stats
    assert stats["count"] == 2
    assert stats["means"][0] == 2.0


def test_merge_feature_stats() -> None:
    existing = {
        "sum": [2.0, 2.0],
        "sum_squares": [4.0, 4.0],
        "mins": [1.0, 1.0],
        "maxs": [1.0, 1.0],
        "count": 2,
    }
    incoming = {
        "sum": [3.0, 3.0],
        "sum_squares": [9.0, 9.0],
        "mins": [1.0, 1.0],
        "maxs": [2.0, 2.0],
        "count": 1,
    }
    merged = merge_feature_stats(existing, incoming)
    assert merged["count"] == 3
    assert merged["maxs"][0] == 2.0


def test_compute_daily_feature_stats_multiple_days() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    next_day = datetime(2024, 1, 2, tzinfo=timezone.utc)
    schema_id = uuid.uuid4()
    epochs = [
        WindowedEpoch(
            epoch_index=0,
            epoch_start_ts=start,
            features=np.array([1.0, 3.0]),
            feature_schema_id=schema_id,
        ),
        WindowedEpoch(
            epoch_index=1,
            epoch_start_ts=start,
            features=np.array([3.0, 5.0]),
            feature_schema_id=schema_id,
        ),
        WindowedEpoch(
            epoch_index=2,
            epoch_start_ts=next_day,
            features=np.array([2.0, 4.0]),
            feature_schema_id=schema_id,
        ),
    ]
    aggregates = compute_daily_feature_stats(epochs)
    assert {item.stat_date for item in aggregates} == {
        start.date(),
        next_day.date(),
    }
    day_one = next(item for item in aggregates if item.stat_date == start.date())
    assert day_one.stats["count"] == 2
    assert day_one.stats["means"] == [2.0, 4.0]
    assert day_one.stats["p50"] == pytest.approx([2.0, 4.0])


def test_extract_means_from_sums() -> None:
    stats = {"sum": [6.0, 10.0], "sum_squares": [0.0, 0.0], "count": 2}
    means = extract_means(stats)
    assert means == [3.0, 5.0]


def test_merge_feature_stats_weighted_percentiles() -> None:
    existing = {
        "sum": [2.0],
        "sum_squares": [4.0],
        "mins": [1.0],
        "maxs": [3.0],
        "p10": [1.0],
        "p50": [2.0],
        "p90": [3.0],
        "count": 2,
    }
    incoming = {
        "sum": [6.0],
        "sum_squares": [20.0],
        "mins": [2.0],
        "maxs": [4.0],
        "p10": [2.0],
        "p50": [3.0],
        "p90": [4.0],
        "count": 1,
    }
    merged = merge_feature_stats(existing, incoming)
    assert merged["p50"] == pytest.approx([2.3333333333])
