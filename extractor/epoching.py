from __future__ import annotations

from typing import Any, Iterable


def epoch_slices(
    num_samples: int, fs: float, epoch_sec: int, n_epochs: int
) -> list[tuple[int, int]]:
    samples_per_epoch = int(round(fs * epoch_sec))
    slices: list[tuple[int, int]] = []
    for i in range(n_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        slices.append((start, end))
    return slices


def epoch_signal(
    data, fs: float, epoch_sec: int, n_epochs: int
) -> Iterable[tuple[int, int, Any]]:
    for start, end in epoch_slices(len(data), fs, epoch_sec, n_epochs):
        if end > len(data):
            yield start, end, None
        else:
            yield start, end, data[start:end]
