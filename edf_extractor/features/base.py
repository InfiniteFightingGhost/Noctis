from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edf_extractor.config import ExtractConfig
from edf_extractor.models import SignalSeries


@dataclass
class FeatureContext:
    config: ExtractConfig
    n_epochs: int
    signals: dict[str, SignalSeries]
    hypnogram: np.ndarray


@dataclass
class FeatureOutput:
    features: dict[str, np.ndarray]
    flags: np.ndarray | None = None
    agree_flags: np.ndarray | None = None
    warnings: list[str] | None = None
    qc: dict[str, float] | None = None


class FeaturePlugin:
    name: str

    def compute(self, ctx: FeatureContext) -> FeatureOutput:
        raise NotImplementedError
