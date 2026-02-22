from __future__ import annotations

import numpy as np

from dreem_extractor.constants import FlagBits, INT8_UNKNOWN, UINT8_UNKNOWN
from dreem_extractor.features.base import FeatureContext, FeatureOutput, FeaturePlugin
from dreem_extractor.features.utils import bandpass_filter, detect_peaks, mad


class RespRRPlugin(FeaturePlugin):
    name = "rr_resp"

    def compute(self, ctx: FeatureContext) -> FeatureOutput:
        n_epochs = ctx.n_epochs
        rr_mean = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
        rr_std = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
        drr = np.full(n_epochs, INT8_UNKNOWN, dtype=np.int8)
        flags = np.zeros(n_epochs, dtype=np.uint8)

        if "resp" not in ctx.signals:
            return FeatureOutput(
                features={"rr_mean": rr_mean, "rr_std": rr_std, "drr": drr},
                flags=flags,
            )

        resp = ctx.signals["resp"]
        flags |= 1 << FlagBits.RESP_PRESENT

        filtered = bandpass_filter(
            resp.data,
            resp.fs,
            ctx.config.thresholds.resp_band_hz[0],
            ctx.config.thresholds.resp_band_hz[1],
        )
        threshold = np.median(filtered) + 0.5 * mad(filtered)
        peaks = detect_peaks(filtered, resp.fs, min_distance_sec=1.0, height=threshold)
        if peaks.size < 2:
            return FeatureOutput(
                features={"rr_mean": rr_mean, "rr_std": rr_std, "drr": drr},
                flags=flags,
            )

        peak_times = peaks / resp.fs
        rr_intervals = np.diff(peak_times)
        valid_mask = (rr_intervals >= 1.0) & (rr_intervals <= 10.0)
        if not np.any(valid_mask):
            return FeatureOutput(
                features={"rr_mean": rr_mean, "rr_std": rr_std, "drr": drr},
                flags=flags,
            )
        rr_intervals = rr_intervals[valid_mask]
        rr_values = 60.0 / rr_intervals
        rr_times = (peak_times[1:])[valid_mask]

        min_rr, max_rr = ctx.config.thresholds.rr_range_bpm
        min_breaths = ctx.config.thresholds.min_breaths_per_epoch

        samples_per_epoch = int(round(resp.fs * ctx.config.epoch_sec))
        for i in range(n_epochs):
            start_t = (i * samples_per_epoch) / resp.fs
            end_t = ((i + 1) * samples_per_epoch) / resp.fs
            mask = (rr_times >= start_t) & (rr_times < end_t)
            if not np.any(mask):
                continue
            values = rr_values[mask]
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
            features={"rr_mean": rr_mean, "rr_std": rr_std, "drr": drr},
            flags=flags,
        )
