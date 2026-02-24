from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, lfilter, welch


def bandpass_filter(
    data: np.ndarray, fs: float, low: float, high: float, order: int = 4
) -> np.ndarray:
    nyq = 0.5 * fs
    low_norm = max(0.0, low / nyq)
    high_norm = min(0.99, high / nyq)
    if high_norm <= 0 or low_norm >= 1 or low_norm >= high_norm:
        return data
    b, a = butter(order, [low_norm, high_norm], btype="band")
    return lfilter(b, a, data)


def lowpass_filter(data: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    norm = cutoff / nyq
    if norm >= 1:
        return data
    b, a = butter(order, norm, btype="low")
    return lfilter(b, a, data)


def mad(data: np.ndarray) -> float:
    median = np.median(data)
    return float(np.median(np.abs(data - median)))


def detect_peaks(
    data: np.ndarray, fs: float, min_distance_sec: float, height: float | None = None
) -> np.ndarray:
    distance = max(1, int(round(min_distance_sec * fs)))
    peaks, _ = find_peaks(data, distance=distance, height=height)
    return peaks


def bandpower_ratio(
    data: np.ndarray,
    fs: float,
    band: tuple[float, float],
    total_band: tuple[float, float],
) -> float:
    if data.size == 0:
        return 0.0
    freqs, power = welch(data, fs=fs, nperseg=min(256, data.size))
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    total_mask = (freqs >= total_band[0]) & (freqs <= total_band[1])
    band_power = float(np.trapz(power[band_mask], freqs[band_mask]))
    total_power = float(np.trapz(power[total_mask], freqs[total_mask]))
    if total_power <= 0:
        return 0.0
    return band_power / total_power
