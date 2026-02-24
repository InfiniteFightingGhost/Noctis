from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SignalSeries:
    name: str
    label: str
    fs: float
    data: np.ndarray
    segments: list[np.ndarray | None]


@dataclass
class RecordManifest:
    record_id: str
    channel_map: dict[str, str]
    fs_map: dict[str, float]
    start_time: str | None


@dataclass
class ExtractResult:
    record_id: str
    hypnogram: np.ndarray
    features: np.ndarray
    feature_mask: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any]
    qc: dict[str, Any]
    warnings: list[str]
