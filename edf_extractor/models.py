from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SignalSeries:
    name: str
    data: np.ndarray
    fs: float
    segments: list[np.ndarray | None]


@dataclass
class RecordManifest:
    record_id: str
    hypnogram_ref: str | None
    start_time: str | None
    paths_present: list[str]
    channel_map: dict[str, str]
    fs_map: dict[str, float]
    channel_index_map: dict[str, int]


@dataclass
class ExtractResult:
    record_id: str
    hypnogram: np.ndarray
    features: np.ndarray
    valid_mask: np.ndarray
    timestamps: np.ndarray | None
    metadata: dict[str, Any]
    qc: dict[str, Any]
    warnings: list[str]
