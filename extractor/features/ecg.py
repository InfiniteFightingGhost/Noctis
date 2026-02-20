from __future__ import annotations

from typing import Any

import numpy as np

from extractor.config import (
    INT8_UNKNOWN,
    UINT8_UNKNOWN,
    ExtractConfig,
    clamp_int8,
    clamp_uint8,
)
from extractor.epoching import epoch_slices
from extractor.features.utils import bandpass_filter, detect_peaks, mad


def compute_ecg_features(
    signal: dict[str, Any] | None,
    n_epochs: int,
    config: ExtractConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if signal is None:
        return _unknown(n_epochs)
    data = signal["data"]
    fs = signal["fs"]
    if fs is None or data.size == 0:
        return _unknown(n_epochs)

    filtered = bandpass_filter(data.astype(float), fs, config.ecg_band_low, config.ecg_band_high)
    threshold = np.median(filtered) + 1.5 * mad(filtered)
    peaks = detect_peaks(filtered, fs, min_distance_sec=0.3, height=threshold)
    if peaks.size < 2:
        return _unknown(n_epochs)

    peak_times = peaks / fs
    rr_intervals = np.diff(peak_times)
    valid_rr_mask = (rr_intervals >= 0.3) & (rr_intervals <= 2.5)
    if not np.any(valid_rr_mask):
        return _unknown(n_epochs)
    rr_intervals = rr_intervals[valid_rr_mask]
    hr_values = 60.0 / rr_intervals
    hr_times = (peak_times[1:])[valid_rr_mask]

    hr_mean = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    hr_std = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    dhr = np.full(n_epochs, INT8_UNKNOWN, dtype=np.int8)
    valid = np.zeros(n_epochs, dtype=bool)

    for i, (start, end) in enumerate(epoch_slices(len(data), fs, config.epoch_sec, n_epochs)):
        start_t = start / fs
        end_t = end / fs
        mask = (hr_times >= start_t) & (hr_times < end_t)
        if not np.any(mask):
            continue
        values = hr_values[mask]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        hr_mean[i] = clamp_uint8(mean_val)
        hr_std[i] = clamp_uint8(std_val)
        valid[i] = True

    hr_values = _to_nan(hr_mean)
    hr_values = _post_process_hr(hr_values, config)
    hr_values = _smooth_hr(hr_values, config.hr_smooth_alpha, config.hr_smooth_window)
    hr_mean = _from_float_to_uint8(hr_values)

    for i in range(1, n_epochs):
        if np.isnan(hr_values[i]) or np.isnan(hr_values[i - 1]):
            dhr[i] = INT8_UNKNOWN
        else:
            dhr[i] = clamp_int8(int(round(hr_values[i] - hr_values[i - 1])))

    return hr_mean, hr_std, dhr, valid


def _unknown(n_epochs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hr_mean = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    hr_std = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    dhr = np.full(n_epochs, INT8_UNKNOWN, dtype=np.int8)
    valid = np.zeros(n_epochs, dtype=bool)
    return hr_mean, hr_std, dhr, valid


def _to_nan(values: np.ndarray) -> np.ndarray:
    out = values.astype(float)
    out[values == UINT8_UNKNOWN] = np.nan
    return out


def _from_float_to_uint8(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, UINT8_UNKNOWN, dtype=np.uint8)
    mask = ~np.isnan(values)
    if np.any(mask):
        out[mask] = np.clip(np.round(values[mask]), 0, 255).astype(np.uint8)
    return out


def _post_process_hr(values: np.ndarray, config: ExtractConfig) -> np.ndarray:
    out = _fill_short_nan_runs(values, config.hr_gap_fill_max_epochs)
    out = _rolling_median(out, config.hr_median_window)
    out = _cap_delta(out, config.hr_jump_max_delta)
    return out


def _fill_short_nan_runs(values: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return values
    out = values.copy()
    idx = 0
    n = len(out)
    while idx < n:
        if not np.isnan(out[idx]):
            idx += 1
            continue
        start = idx
        while idx < n and np.isnan(out[idx]):
            idx += 1
        end = idx - 1
        gap_len = end - start + 1
        prev_idx = start - 1
        next_idx = idx if idx < n else None
        if (
            gap_len <= max_gap
            and prev_idx >= 0
            and next_idx is not None
            and not np.isnan(out[prev_idx])
            and not np.isnan(out[next_idx])
        ):
            fill = np.linspace(out[prev_idx], out[next_idx], gap_len + 2)[1:-1]
            out[start:idx] = fill
    return out


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    out = values.copy()
    half = window // 2
    n = len(out)
    for i in range(n):
        if np.isnan(out[i]):
            continue
        start = max(0, i - half)
        end = min(n, i + half + 1)
        median_val = np.nanmedian(out[start:end])
        if not np.isnan(median_val):
            out[i] = median_val
    return out


def _cap_delta(values: np.ndarray, max_delta: float) -> np.ndarray:
    if max_delta <= 0:
        return values
    out = values.copy()
    prev = None
    for i in range(len(out)):
        if np.isnan(out[i]):
            prev = None
            continue
        if prev is None:
            prev = float(out[i])
            continue
        delta = float(out[i]) - prev
        if abs(delta) > max_delta:
            out[i] = prev + np.sign(delta) * max_delta
        prev = float(out[i])
    return out


def _smooth_hr(values: np.ndarray, alpha: float, window: int) -> np.ndarray:
    out = values.copy()
    if alpha > 0:
        prev = None
        for i in range(len(out)):
            if np.isnan(out[i]):
                prev = None
                continue
            if prev is None:
                prev = float(out[i])
                continue
            out[i] = alpha * float(out[i]) + (1.0 - alpha) * prev
            prev = float(out[i])
    if window <= 1:
        return out
    if window % 2 == 0:
        window += 1
    smoothed = out.copy()
    half = window // 2
    n = len(out)
    for i in range(n):
        if np.isnan(out[i]):
            continue
        start = max(0, i - half)
        end = min(n, i + half + 1)
        segment = out[start:end]
        if np.all(np.isnan(segment)):
            continue
        smoothed[i] = float(np.nanmean(segment))
    return smoothed
