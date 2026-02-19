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
from extractor.features.utils import bandpass_filter, bandpower_ratio, detect_peaks, mad


def compute_resp_features(
    signal: dict[str, Any] | None,
    ecg_signal: dict[str, Any] | None,
    n_epochs: int,
    config: ExtractConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if signal is None:
        if ecg_signal is None:
            return _unknown(n_epochs)
        edr = compute_edr(ecg_signal, config)
        if edr is None:
            return _unknown(n_epochs)
        return _resp_from_signal(edr, n_epochs, config)
    return _resp_from_signal(signal, n_epochs, config)


def _resp_from_signal(
    signal: dict[str, Any],
    n_epochs: int,
    config: ExtractConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = signal["data"]
    fs = signal["fs"]
    if fs is None or data.size == 0:
        return _unknown(n_epochs)

    filtered = bandpass_filter(
        data.astype(float), fs, config.resp_band_low, config.resp_band_high
    )
    threshold = np.median(filtered) + 0.5 * mad(filtered)
    peaks = detect_peaks(filtered, fs, min_distance_sec=1.0, height=threshold)
    if peaks.size < 2:
        return _unknown(n_epochs)

    peak_times = peaks / fs
    rr_intervals = np.diff(peak_times)
    valid_rr_mask = (rr_intervals >= 1.0) & (rr_intervals <= 10.0)
    if not np.any(valid_rr_mask):
        return _unknown(n_epochs)
    rr_intervals = rr_intervals[valid_rr_mask]
    rr_values = 60.0 / rr_intervals
    rr_times = (peak_times[1:])[valid_rr_mask]

    rr_mean = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    rr_std = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    drr = np.full(n_epochs, INT8_UNKNOWN, dtype=np.int8)
    valid = np.zeros(n_epochs, dtype=bool)

    for i, (start, end) in enumerate(
        epoch_slices(len(data), fs, config.epoch_sec, n_epochs)
    ):
        start_t = start / fs
        end_t = end / fs
        mask = (rr_times >= start_t) & (rr_times < end_t)
        if not np.any(mask):
            continue
        values = rr_values[mask]
        rr_mean[i] = clamp_uint8(float(np.mean(values)))
        rr_std[i] = clamp_uint8(float(np.std(values)))
        valid[i] = True

    for i in range(1, n_epochs):
        if rr_mean[i] == UINT8_UNKNOWN or rr_mean[i - 1] == UINT8_UNKNOWN:
            drr[i] = INT8_UNKNOWN
        else:
            drr[i] = clamp_int8(int(rr_mean[i]) - int(rr_mean[i - 1]))

    return rr_mean, rr_std, drr, valid


def compute_edr(signal: dict[str, Any], config: ExtractConfig) -> dict[str, Any] | None:
    data = signal["data"]
    fs = signal["fs"]
    if fs is None or data.size == 0:
        return None
    filtered = bandpass_filter(
        data.astype(float), fs, config.ecg_band_low, config.ecg_band_high
    )
    threshold = np.median(filtered) + 1.5 * mad(filtered)
    peaks = detect_peaks(filtered, fs, min_distance_sec=0.3, height=threshold)
    if peaks.size < 5:
        return None
    peak_times = peaks / fs
    peak_amp = filtered[peaks]
    target_fs = 4.0
    duration = len(data) / fs
    times = np.arange(0, duration, 1.0 / target_fs)
    interp = np.interp(
        times, peak_times, peak_amp, left=peak_amp[0], right=peak_amp[-1]
    )
    return {"data": interp, "fs": target_fs}


def compute_resp_quality(
    signal: dict[str, Any] | None,
    n_epochs: int,
    config: ExtractConfig,
) -> np.ndarray:
    if signal is None:
        return np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    data = signal["data"]
    fs = signal["fs"]
    if fs is None or data.size == 0:
        return np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    quality = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    for i, (start, end) in enumerate(
        epoch_slices(len(data), fs, config.epoch_sec, n_epochs)
    ):
        if end > len(data):
            continue
        segment = data[start:end]
        ratio = bandpower_ratio(
            segment.astype(float),
            fs,
            band=(config.resp_band_low, config.resp_band_high),
            total_band=(0.05, 2.0),
        )
        score = int(round(min(100.0, max(0.0, ratio * 100.0))))
        quality[i] = score
    return quality


def _unknown(n_epochs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rr_mean = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    rr_std = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    drr = np.full(n_epochs, INT8_UNKNOWN, dtype=np.int8)
    valid = np.zeros(n_epochs, dtype=bool)
    return rr_mean, rr_std, drr, valid
