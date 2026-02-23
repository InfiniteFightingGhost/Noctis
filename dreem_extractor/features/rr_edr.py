from __future__ import annotations

import numpy as np

from dreem_extractor.constants import FlagBits, INT8_UNKNOWN, UINT8_UNKNOWN
from dreem_extractor.features.base import FeatureContext, FeatureOutput, FeaturePlugin
from dreem_extractor.features.utils import (
    bandpass_filter,
    bandpower_ratio,
    detect_peaks,
    mad,
)


class EDRPlugin(FeaturePlugin):
    name = "rr_edr"

    def compute(self, ctx: FeatureContext) -> FeatureOutput:
        n_epochs = ctx.n_epochs
        rr_mean = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
        rr_std = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
        drr = np.full(n_epochs, INT8_UNKNOWN, dtype=np.int8)
        vib_resp_q = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
        flags = np.zeros(n_epochs, dtype=np.uint8)

        if "ecg" not in ctx.signals:
            return FeatureOutput(
                features={
                    "rr_mean": rr_mean,
                    "rr_std": rr_std,
                    "drr": drr,
                    "vib_resp_q": vib_resp_q,
                },
                flags=flags,
            )

        ecg = ctx.signals["ecg"]
        flags |= 1 << FlagBits.RR_FROM_EDR

        filtered = bandpass_filter(
            ecg.data,
            ecg.fs,
            ctx.config.thresholds.ecg_band_hz[0],
            ctx.config.thresholds.ecg_band_hz[1],
        )
        threshold = np.median(filtered) + 1.5 * mad(filtered)
        peaks = detect_peaks(filtered, ecg.fs, min_distance_sec=0.3, height=threshold)
        if peaks.size < 5:
            return FeatureOutput(
                features={
                    "rr_mean": rr_mean,
                    "rr_std": rr_std,
                    "drr": drr,
                    "vib_resp_q": vib_resp_q,
                },
                flags=flags,
            )

        peak_times = peaks / ecg.fs
        peak_amp = filtered[peaks]
        edr_fs = 4.0
        duration = len(ecg.data) / ecg.fs
        times = np.arange(0, duration, 1.0 / edr_fs)
        edr = np.interp(
            times, peak_times, peak_amp, left=peak_amp[0], right=peak_amp[-1]
        )

        edr_filtered = bandpass_filter(
            edr,
            edr_fs,
            ctx.config.thresholds.edr_band_hz[0],
            ctx.config.thresholds.edr_band_hz[1],
        )
        min_rr, max_rr = ctx.config.thresholds.rr_range_bpm
        min_breaths = ctx.config.thresholds.min_breaths_per_epoch

        samples_per_epoch = int(round(edr_fs * ctx.config.epoch_sec))
        for i in range(n_epochs):
            start = i * samples_per_epoch
            end = start + samples_per_epoch
            if end > edr_filtered.size:
                continue
            segment = edr_filtered[start:end]
            ratio = bandpower_ratio(
                segment,
                edr_fs,
                band=ctx.config.thresholds.edr_band_hz,
                total_band=(0.05, 2.0),
            )
            quality = int(round(min(255.0, max(0.0, ratio * 255.0))))
            vib_resp_q[i] = quality
            if quality < ctx.config.thresholds.rr_quality_min:
                continue
            peaks = detect_peaks(segment, edr_fs, min_distance_sec=1.0)
            if peaks.size < 2:
                continue
            peak_times = peaks / edr_fs
            rr_intervals = np.diff(peak_times)
            valid_mask = (rr_intervals >= 1.0) & (rr_intervals <= 10.0)
            if not np.any(valid_mask):
                continue
            rr_intervals = rr_intervals[valid_mask]
            values = 60.0 / rr_intervals
            if values.size < min_breaths:
                continue
            mean_val = float(np.mean(values))
            if mean_val < min_rr or mean_val > max_rr:
                continue
            rr_mean[i] = int(round(min(255, max(0, mean_val))))
            rr_std[i] = int(round(min(255, max(0, float(np.std(values))))))
            flags[i] |= 1 << FlagBits.RR_VALID

        for i in range(1, n_epochs):
            if rr_mean[i] == UINT8_UNKNOWN or rr_mean[i - 1] == UINT8_UNKNOWN:
                drr[i] = INT8_UNKNOWN
            else:
                delta = int(rr_mean[i]) - int(rr_mean[i - 1])
                drr[i] = int(max(-128, min(127, delta)))

        return FeatureOutput(
            features={
                "rr_mean": rr_mean,
                "rr_std": rr_std,
                "drr": drr,
                "vib_resp_q": vib_resp_q,
            },
            flags=flags,
        )
