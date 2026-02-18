from __future__ import annotations

import numpy as np


def align_signal(
    data: np.ndarray,
    fs: float,
    n_epochs: int,
    epoch_sec: int,
) -> tuple[list[np.ndarray | None], list[str]]:
    warnings: list[str] = []
    samples_per_epoch = int(round(fs * epoch_sec))
    expected = samples_per_epoch * n_epochs
    if data.size < expected:
        warnings.append("signal_shorter_than_hypnogram")
    if data.size > expected:
        warnings.append("signal_longer_than_hypnogram")
    segments: list[np.ndarray | None] = []
    for i in range(n_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        if end > data.size:
            segments.append(None)
        else:
            segments.append(data[start:end])
    return segments, warnings
