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

    filtered = bandpass_filter(
        data.astype(float), fs, config.ecg_band_low, config.ecg_band_high
    )
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

    for i, (start, end) in enumerate(
        epoch_slices(len(data), fs, config.epoch_sec, n_epochs)
    ):
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

    for i in range(1, n_epochs):
        if hr_mean[i] == UINT8_UNKNOWN or hr_mean[i - 1] == UINT8_UNKNOWN:
            dhr[i] = INT8_UNKNOWN
        else:
            dhr[i] = clamp_int8(int(hr_mean[i]) - int(hr_mean[i - 1]))

    return hr_mean, hr_std, dhr, valid


def _unknown(n_epochs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hr_mean = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    hr_std = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    dhr = np.full(n_epochs, INT8_UNKNOWN, dtype=np.int8)
    valid = np.zeros(n_epochs, dtype=bool)
    return hr_mean, hr_std, dhr, valid
