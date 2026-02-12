from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import pi

import numpy as np


@dataclass(frozen=True)
class SyntheticFeatureGenerator:
    feature_size: int
    seed: int

    def generate_epoch_batch(
        self,
        *,
        device_index: int,
        start_ts: datetime,
        epoch_seconds: int,
        start_index: int,
        count: int,
        total_epochs: int,
        feature_schema_version: str,
    ) -> list[dict[str, object]]:
        base_rng = np.random.default_rng(self.seed + device_index)
        base_mean = base_rng.normal(0.0, 0.5, self.feature_size)
        amplitude = base_rng.normal(0.15, 0.05, self.feature_size)
        phase = base_rng.random(self.feature_size) * 2 * pi

        epochs: list[dict[str, object]] = []
        for idx in range(start_index, start_index + count):
            t = idx / max(1, total_epochs)
            signal = np.sin(2 * pi * t + phase) * amplitude
            noise_rng = np.random.default_rng(self.seed + device_index + idx)
            noise = noise_rng.normal(0.0, 0.05, self.feature_size)
            features = (base_mean + signal + noise).astype(np.float32)
            epochs.append(
                {
                    "epoch_index": idx,
                    "epoch_start_ts": start_ts + timedelta(seconds=epoch_seconds * idx),
                    "feature_schema_version": feature_schema_version,
                    "features": features.tolist(),
                }
            )
        return epochs
