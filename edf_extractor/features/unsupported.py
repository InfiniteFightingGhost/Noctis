from __future__ import annotations

import numpy as np

from edf_extractor.constants import FlagBits, UINT8_UNKNOWN
from edf_extractor.features.base import FeatureContext, FeatureOutput, FeaturePlugin


class UnsupportedPlugin(FeaturePlugin):
    name = "unsupported"

    def compute(self, ctx: FeatureContext) -> FeatureOutput:
        n_epochs = ctx.n_epochs
        features = {
            "in_bed_pct": np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8),
            "large_move_pct": np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8),
            "minor_move_pct": np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8),
            "turnovers_delta": np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8),
            "apnea_delta": np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8),
            "vib_move_pct": np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8),
            "agree_flags": np.zeros(n_epochs, dtype=np.uint8),
        }
        flags = np.zeros(n_epochs, dtype=np.uint8)
        if ctx.config.flags.get("unsupported_fields", True):
            flags |= 1 << FlagBits.UNSUPPORTED_FIELDS
        return FeatureOutput(features=features, flags=flags)
