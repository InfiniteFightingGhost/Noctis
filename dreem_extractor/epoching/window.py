from __future__ import annotations


def epoch_indices(n_epochs: int, epoch_sec: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n_epochs)]
