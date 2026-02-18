from __future__ import annotations

from typing import Any

import numpy as np

from extractor.config import UINT8_UNKNOWN, ExtractConfig
from extractor.epoching import epoch_slices
from extractor.features.utils import lowpass_filter, mad


def compute_movement_features(
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

    envelope = np.abs(data.astype(float))
    envelope = lowpass_filter(envelope, fs, cutoff=5.0)

    median = np.median(envelope)
    mad_val = mad(envelope)
    if mad_val == 0:
        mad_val = float(np.std(envelope))
    minor_th = median + config.minor_move_k * mad_val
    large_th = median + config.large_move_k * mad_val

    minor_pct = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    large_pct = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    burst_counts = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    turnovers = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    valid = np.zeros(n_epochs, dtype=bool)

    for i, (start, end) in enumerate(
        epoch_slices(len(data), fs, config.epoch_sec, n_epochs)
    ):
        if end > len(data):
            continue
        segment = envelope[start:end]
        if segment.size == 0:
            continue
        minor_mask = segment > minor_th
        large_mask = segment > large_th
        minor_pct[i] = int(round(min(255.0, max(0.0, 100.0 * np.mean(minor_mask)))))
        large_pct[i] = int(round(min(255.0, max(0.0, 100.0 * np.mean(large_mask)))))
        burst_counts[i] = int(round(min(255.0, count_bursts(large_mask))))
        valid[i] = True

    prev = None
    for i in range(n_epochs):
        if burst_counts[i] == UINT8_UNKNOWN:
            prev = None
            continue
        if prev is None:
            turnovers[i] = UINT8_UNKNOWN
            prev = int(burst_counts[i])
            continue
        delta = int(burst_counts[i]) - prev
        turnovers[i] = int(round(min(255, max(0, delta))))
        prev = int(burst_counts[i])

    return large_pct, minor_pct, turnovers, valid


def count_bursts(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    diffs = np.diff(mask.astype(int), prepend=0)
    return int(np.sum(diffs == 1))


def _unknown(n_epochs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    large_pct = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    minor_pct = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    turnovers = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    valid = np.zeros(n_epochs, dtype=bool)
    return large_pct, minor_pct, turnovers, valid
