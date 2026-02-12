from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import sqrt
from typing import Iterable, cast

import numpy as np

from app.services.windowing import WindowedEpoch


@dataclass(frozen=True)
class FeatureAggregate:
    stat_date: date
    stats: dict[str, object]


def compute_daily_feature_stats(
    epochs: Iterable[WindowedEpoch],
) -> list[FeatureAggregate]:
    buckets: dict[date, list[np.ndarray]] = {}
    for epoch in epochs:
        bucket = epoch.epoch_start_ts.date()
        buckets.setdefault(bucket, []).append(epoch.features)

    aggregates: list[FeatureAggregate] = []
    for stat_date, rows in buckets.items():
        matrix = np.stack(rows, axis=0)
        sums = matrix.sum(axis=0)
        sum_squares = np.square(matrix).sum(axis=0)
        mins = matrix.min(axis=0)
        maxs = matrix.max(axis=0)
        percentiles = np.percentile(matrix, [10, 50, 90], axis=0)
        count = int(matrix.shape[0])
        means, stds = _compute_mean_std(sums, sum_squares, count)
        stats = {
            "sum": sums.tolist(),
            "sum_squares": sum_squares.tolist(),
            "mins": mins.tolist(),
            "maxs": maxs.tolist(),
            "count": count,
            "means": means,
            "stds": stds,
            "p10": percentiles[0].tolist(),
            "p50": percentiles[1].tolist(),
            "p90": percentiles[2].tolist(),
        }
        aggregates.append(FeatureAggregate(stat_date=stat_date, stats=stats))
    return aggregates


def merge_feature_stats(
    existing: dict[str, object], incoming: dict[str, object]
) -> dict[str, object]:
    existing_norm: dict[str, list[float] | int] = _normalize_stats(existing)
    incoming_norm: dict[str, list[float] | int] = _normalize_stats(incoming)

    existing_sum = cast(list[float], existing_norm["sum"])
    incoming_sum = cast(list[float], incoming_norm["sum"])
    existing_sum_squares = cast(list[float], existing_norm["sum_squares"])
    incoming_sum_squares = cast(list[float], incoming_norm["sum_squares"])
    existing_mins = cast(list[float], existing_norm["mins"])
    incoming_mins = cast(list[float], incoming_norm["mins"])
    existing_maxs = cast(list[float], existing_norm["maxs"])
    incoming_maxs = cast(list[float], incoming_norm["maxs"])
    existing_p10 = cast(list[float], existing_norm["p10"])
    incoming_p10 = cast(list[float], incoming_norm["p10"])
    existing_p50 = cast(list[float], existing_norm["p50"])
    incoming_p50 = cast(list[float], incoming_norm["p50"])
    existing_p90 = cast(list[float], existing_norm["p90"])
    incoming_p90 = cast(list[float], incoming_norm["p90"])
    existing_count = int(existing_norm["count"])
    incoming_count = int(incoming_norm["count"])

    sums = _merge_lists(existing_sum, incoming_sum, lambda a, b: a + b)
    sum_squares = _merge_lists(
        existing_sum_squares,
        incoming_sum_squares,
        lambda a, b: a + b,
    )
    mins = _merge_lists(existing_mins, incoming_mins, min)
    maxs = _merge_lists(existing_maxs, incoming_maxs, max)
    count = existing_count + incoming_count
    p10 = _merge_weighted_lists(
        existing_p10,
        incoming_p10,
        existing_count,
        incoming_count,
    )
    p50 = _merge_weighted_lists(
        existing_p50,
        incoming_p50,
        existing_count,
        incoming_count,
    )
    p90 = _merge_weighted_lists(
        existing_p90,
        incoming_p90,
        existing_count,
        incoming_count,
    )
    means, stds = _compute_mean_std(np.array(sums), np.array(sum_squares), count)
    return {
        "sum": sums,
        "sum_squares": sum_squares,
        "mins": mins,
        "maxs": maxs,
        "count": count,
        "means": means,
        "stds": stds,
        "p10": p10,
        "p50": p50,
        "p90": p90,
    }


def extract_means(stats: dict[str, object]) -> list[float]:
    if "means" in stats:
        return [float(value) for value in stats.get("means", [])]
    normalized = _normalize_stats(stats)
    count = int(normalized["count"])
    if count <= 0:
        return []
    sums = cast(list[float], normalized["sum"])
    return [value / count for value in sums]


def _normalize_stats(stats: dict[str, object]) -> dict[str, list[float] | int]:
    if "sum" in stats and "sum_squares" in stats:
        count = int(stats.get("count", 0) or 0)
        sums = _extract_float_list(stats, "sum")
        sum_squares = _extract_float_list(stats, "sum_squares")
        means = (
            [value / count for value in sums]
            if count > 0 and sums
            else _extract_float_list(stats, "means")
        )
        return {
            "sum": sums,
            "sum_squares": sum_squares,
            "mins": _extract_float_list(stats, "mins"),
            "maxs": _extract_float_list(stats, "maxs"),
            "count": count,
            "p10": _normalize_percentiles(stats, "p10", means),
            "p50": _normalize_percentiles(stats, "p50", means),
            "p90": _normalize_percentiles(stats, "p90", means),
        }
    count = int(stats.get("count", 0) or 0)
    means = _extract_float_list(stats, "means")
    stds = _extract_float_list(stats, "stds")
    if len(stds) < len(means):
        stds = [*stds, *([0.0] * (len(means) - len(stds)))]
    sums = [value * count for value in means]
    sum_squares = [
        ((stds[idx] ** 2) + (means[idx] ** 2)) * count for idx in range(len(means))
    ]
    mins = _extract_float_list(stats, "mins", fallback=means)
    maxs = _extract_float_list(stats, "maxs", fallback=means)
    return {
        "sum": sums,
        "sum_squares": sum_squares,
        "mins": mins,
        "maxs": maxs,
        "count": count,
        "p10": _normalize_percentiles(stats, "p10", means),
        "p50": _normalize_percentiles(stats, "p50", means),
        "p90": _normalize_percentiles(stats, "p90", means),
    }


def _compute_mean_std(
    sums: np.ndarray,
    sum_squares: np.ndarray,
    count: int,
) -> tuple[list[float], list[float]]:
    if count <= 0:
        return [], []
    means = (sums / count).tolist()
    variances = (sum_squares / count) - np.square(sums / count)
    variances = np.maximum(variances, 0.0)
    stds = [sqrt(float(value)) for value in variances]
    return [float(value) for value in means], stds


def _merge_lists(
    left: list[float],
    right: list[float],
    op,
) -> list[float]:
    size = min(len(left), len(right))
    merged = [float(op(left[idx], right[idx])) for idx in range(size)]
    return merged


def _merge_weighted_lists(
    left: list[float],
    right: list[float],
    left_count: int,
    right_count: int,
) -> list[float]:
    total = left_count + right_count
    if total <= 0:
        return []
    size = min(len(left), len(right))
    merged = [
        float((left[idx] * left_count + right[idx] * right_count) / total)
        for idx in range(size)
    ]
    return merged


def _normalize_percentiles(
    stats: dict[str, object], key: str, fallback: list[float]
) -> list[float]:
    values = stats.get(key)
    if isinstance(values, list):
        return [float(value) for value in values]
    return [float(value) for value in fallback]


def _extract_float_list(
    stats: dict[str, object], key: str, fallback: list[float] | None = None
) -> list[float]:
    values = stats.get(key)
    if isinstance(values, list):
        return [float(value) for value in values]
    if fallback is None:
        return []
    return [float(value) for value in fallback]
