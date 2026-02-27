from __future__ import annotations

from typing import Any

import numpy as np


def build_night_summary(labels: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
    valid_labels = labels[valid_mask]
    stage_counts = {str(stage): int(np.sum(valid_labels == stage)) for stage in (-1, 0, 1, 2, 3, 4)}
    return {
        "n_epochs": int(labels.shape[0]),
        "valid_epochs": int(np.sum(valid_mask)),
        "valid_ratio": float(np.mean(valid_mask)) if valid_mask.size else 0.0,
        "stage_counts": stage_counts,
    }
