from __future__ import annotations

import numpy as np

from dreem_extractor.constants import FEATURE_ORDER, FlagBits, UINT8_UNKNOWN
from dreem_extractor.models import ExtractResult, RecordManifest


def compute_qc(result: ExtractResult, manifest: RecordManifest) -> dict[str, object]:
    flags = result.features[:, FEATURE_ORDER.index("flags")]
    hr_valid = (flags & (1 << FlagBits.HR_VALID)) > 0
    rr_valid = (flags & (1 << FlagBits.RR_VALID)) > 0

    hr_vals = result.features[:, FEATURE_ORDER.index("hr_mean")].astype(float)
    rr_vals = result.features[:, FEATURE_ORDER.index("rr_mean")].astype(float)
    hr_vals = hr_vals[hr_vals != UINT8_UNKNOWN]
    rr_vals = rr_vals[rr_vals != UINT8_UNKNOWN]

    def summary(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"min": float("nan"), "max": float("nan"), "median": float("nan")}
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

    missing_channels = []
    for key in ("ecg", "resp"):
        if key not in manifest.channel_map:
            missing_channels.append(key)

    rr_source = "none"
    if np.any((flags & (1 << FlagBits.RR_FROM_EDR)) > 0):
        rr_source = "edr"
    elif "resp" in manifest.channel_map:
        rr_source = "resp"

    return {
        "record_id": result.record_id,
        "missing_channels": missing_channels,
        "hr_valid_pct": float(np.mean(hr_valid)) if hr_valid.size else 0.0,
        "rr_valid_pct": float(np.mean(rr_valid)) if rr_valid.size else 0.0,
        "rr_source": rr_source,
        "alignment_warnings": result.warnings,
        "hr_summary": summary(hr_vals),
        "rr_summary": summary(rr_vals),
    }
