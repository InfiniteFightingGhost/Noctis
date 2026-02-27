from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from extractor_hardened.errors import ExtractionError


@dataclass
class ChannelQC:
    channel: str
    passed: bool
    checks: dict[str, float | bool]
    repairs: list[str]


def run_qc(
    signals: dict[str, Any],
    fs_map: dict[str, float],
    qc_policy: dict[str, Any],
) -> tuple[dict[str, ChannelQC], dict[str, Any]]:
    checks = qc_policy.get("checks", {})
    required = set(qc_policy.get("required_channels", []))
    by_channel: dict[str, ChannelQC] = {}

    for name, series in signals.items():
        data = np.asarray(series.data if hasattr(series, "data") else series)
        fs = float(fs_map.get(name, 0.0))
        nan_inf_fraction = float(np.mean(~np.isfinite(data))) if data.size else 1.0
        finite = data[np.isfinite(data)]
        variance = float(np.var(finite)) if finite.size else 0.0
        diffs = np.diff(finite) if finite.size > 1 else np.array([], dtype=float)
        flatline_fraction = float(np.mean(np.abs(diffs) < 1e-12)) if diffs.size else 1.0
        clip_abs = np.max(np.abs(finite)) if finite.size else 0.0
        clip_ref = np.percentile(np.abs(finite), 99.5) if finite.size else 0.0
        clipping_fraction = float(np.mean(np.abs(finite) >= clip_ref)) if finite.size else 1.0

        passed = True
        if nan_inf_fraction > float(checks.get("nan_inf_fraction_max", 0.02)):
            passed = False
        if flatline_fraction > float(checks.get("flatline_fraction_max", 0.9)):
            passed = False
        if variance < float(checks.get("variance_min", 1e-8)):
            passed = False
        if clipping_fraction > float(checks.get("clipping_fraction_max", 0.3)):
            passed = False

        if name == "ecg":
            ecg_abs_max = float(checks.get("ecg_range_abs_max", 10000.0))
            if clip_abs > ecg_abs_max:
                passed = False
            peak_count = int(np.sum((finite[1:-1] > finite[:-2]) & (finite[1:-1] > finite[2:])))
            if peak_count < int(checks.get("ecg_min_peak_count", 5)):
                passed = False

        eeg_ratio = None
        if name == "eeg" and fs > 0:
            low_band = checks.get("eeg_psd_low_band_hz", [0.5, 4.0])
            high_band = checks.get("eeg_psd_high_band_hz", [20.0, 40.0])
            freqs = np.fft.rfftfreq(finite.size, d=1.0 / fs) if finite.size else np.array([])
            psd = (
                np.abs(np.fft.rfft(finite - np.mean(finite))) ** 2 if finite.size else np.array([])
            )
            if freqs.size and psd.size:
                low_power = float(
                    np.sum(psd[(freqs >= float(low_band[0])) & (freqs <= float(low_band[1]))])
                )
                high_power = float(
                    np.sum(psd[(freqs >= float(high_band[0])) & (freqs <= float(high_band[1]))])
                )
                eeg_ratio = low_power / (high_power + 1e-9)
                if eeg_ratio < float(checks.get("eeg_psd_ratio_min", 0.01)):
                    passed = False

        by_channel[name] = ChannelQC(
            channel=name,
            passed=passed,
            checks={
                "nan_inf_fraction": nan_inf_fraction,
                "variance": variance,
                "flatline_fraction": flatline_fraction,
                "clipping_fraction": clipping_fraction,
                "eeg_psd_ratio": float(eeg_ratio) if eeg_ratio is not None else -1.0,
            },
            repairs=[],
        )

    failed_required = [
        name for name in required if name in by_channel and not by_channel[name].passed
    ]
    if failed_required:
        raise ExtractionError(
            code="E_QC_FAIL",
            message="Required channel failed QC",
            details={"failed_required_channels": failed_required},
        )

    summary = {
        "required_channels": sorted(required),
        "failed_required_channels": failed_required,
        "optional_failed_channels": sorted(
            [
                name
                for name, report in by_channel.items()
                if name not in required and not report.passed
            ]
        ),
    }
    return by_channel, summary
