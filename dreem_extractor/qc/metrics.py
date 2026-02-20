from __future__ import annotations

import numpy as np

from dreem_extractor.config import ExtractConfig
from dreem_extractor.constants import FEATURE_ORDER, FlagBits, UINT8_UNKNOWN
from dreem_extractor.models import ExtractResult, RecordManifest


def compute_qc(
    result: ExtractResult,
    manifest: RecordManifest,
    config: ExtractConfig | None = None,
    extra_metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    flags = result.features[:, FEATURE_ORDER.index("flags")]
    hr_valid = (flags & (1 << FlagBits.HR_VALID)) > 0
    rr_valid = (flags & (1 << FlagBits.RR_VALID)) > 0

    hr_series = result.features[:, FEATURE_ORDER.index("hr_mean")].astype(float)
    rr_series = result.features[:, FEATURE_ORDER.index("rr_mean")].astype(float)
    hr_vals = hr_series[hr_series != UINT8_UNKNOWN]
    rr_vals = rr_series[rr_series != UINT8_UNKNOWN]

    def summary(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"min": float("nan"), "max": float("nan"), "median": float("nan")}
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

    def delta_values(values: np.ndarray) -> np.ndarray:
        deltas: list[float] = []
        prev = None
        for value in values:
            if value == UINT8_UNKNOWN:
                prev = None
                continue
            if prev is None:
                prev = float(value)
                continue
            deltas.append(abs(float(value) - prev))
            prev = float(value)
        return np.array(deltas, dtype=float)

    hr_delta_vals = delta_values(result.features[:, FEATURE_ORDER.index("hr_mean")])
    hr_delta_p95 = float("nan")
    if hr_delta_vals.size:
        hr_delta_p95 = float(np.percentile(hr_delta_vals, 95))
    hr_jump_rate = float("nan")
    if config is not None and hr_delta_vals.size:
        max_delta = config.thresholds.hr_jump_max_delta
        if max_delta > 0:
            hr_jump_rate = float(np.mean(hr_delta_vals > max_delta))
        else:
            hr_jump_rate = 0.0

    missing_channels = []
    for key in ("ecg", "resp"):
        if key not in manifest.channel_map:
            missing_channels.append(key)

    rr_source = "none"
    if np.any((flags & (1 << FlagBits.RR_FROM_EDR)) > 0):
        rr_source = "edr"
    elif "resp" in manifest.channel_map:
        rr_source = "resp"

    qc: dict[str, object] = {
        "record_id": result.record_id,
        "missing_channels": missing_channels,
        "hr_valid_pct": float(np.mean(hr_valid)) if hr_valid.size else 0.0,
        "hr_missing_pct": float(1.0 - np.mean(hr_valid)) if hr_valid.size else 0.0,
        "rr_valid_pct": float(np.mean(rr_valid)) if rr_valid.size else 0.0,
        "rr_source": rr_source,
        "alignment_warnings": result.warnings,
        "hr_summary": summary(hr_vals),
        "rr_summary": summary(rr_vals),
        "hr_jump_rate": hr_jump_rate,
        "hr_delta_p95": hr_delta_p95,
        "hr_smooth_delta_p95": hr_delta_p95,
    }
    if extra_metrics:
        qc.update(extra_metrics)
    return qc
