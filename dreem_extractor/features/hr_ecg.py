from __future__ import annotations

import numpy as np

from dreem_extractor.constants import FlagBits, INT8_UNKNOWN, UINT8_UNKNOWN
from dreem_extractor.features.base import FeatureContext, FeatureOutput, FeaturePlugin
from dreem_extractor.features.utils import bandpass_filter, detect_peaks, mad


class ECGHRPlugin(FeaturePlugin):
    name = "hr_ecg"

    def compute(self, ctx: FeatureContext) -> FeatureOutput:
        n_epochs = ctx.n_epochs
        hr_mean = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
        hr_std = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
        dhr = np.full(n_epochs, INT8_UNKNOWN, dtype=np.int8)
        flags = np.zeros(n_epochs, dtype=np.uint8)

        if "ecg" not in ctx.signals:
            return FeatureOutput(
                features={"hr_mean": hr_mean, "hr_std": hr_std, "dhr": dhr},
                flags=flags,
            )

        ecg = ctx.signals["ecg"]
        flags |= 1 << FlagBits.ECG_PRESENT

        filtered = bandpass_filter(
            ecg.data,
            ecg.fs,
            ctx.config.thresholds.ecg_band_hz[0],
            ctx.config.thresholds.ecg_band_hz[1],
        )
        threshold = np.median(filtered) + 1.5 * mad(filtered)
        peaks = detect_peaks(filtered, ecg.fs, min_distance_sec=0.3, height=threshold)
        if peaks.size < 2:
            return FeatureOutput(
                features={"hr_mean": hr_mean, "hr_std": hr_std, "dhr": dhr},
                flags=flags,
            )

        peak_times = peaks / ecg.fs
        rr_intervals = np.diff(peak_times)
        valid_mask = (rr_intervals >= 0.3) & (rr_intervals <= 2.5)
        if not np.any(valid_mask):
            return FeatureOutput(
                features={"hr_mean": hr_mean, "hr_std": hr_std, "dhr": dhr},
                flags=flags,
            )
        rr_intervals = rr_intervals[valid_mask]
        hr_values = 60.0 / rr_intervals
        hr_times = (peak_times[1:])[valid_mask]

        min_hr, max_hr = ctx.config.thresholds.hr_range_bpm
        min_beats = ctx.config.thresholds.min_beats_per_epoch

        samples_per_epoch = int(round(ecg.fs * ctx.config.epoch_sec))
        for i in range(n_epochs):
            start_t = (i * samples_per_epoch) / ecg.fs
            end_t = ((i + 1) * samples_per_epoch) / ecg.fs
            mask = (hr_times >= start_t) & (hr_times < end_t)
            if not np.any(mask):
                continue
            values = hr_values[mask]
            if values.size < min_beats:
                continue
            mean_val = float(np.mean(values))
            if mean_val < min_hr or mean_val > max_hr:
                continue
            hr_mean[i] = int(round(min(255, max(0, mean_val))))
            hr_std[i] = int(round(min(255, max(0, float(np.std(values))))))
            flags[i] |= 1 << FlagBits.HR_VALID

        for i in range(1, n_epochs):
            if hr_mean[i] == UINT8_UNKNOWN or hr_mean[i - 1] == UINT8_UNKNOWN:
                dhr[i] = INT8_UNKNOWN
            else:
                delta = int(hr_mean[i]) - int(hr_mean[i - 1])
                dhr[i] = int(max(-128, min(127, delta)))

        return FeatureOutput(
            features={"hr_mean": hr_mean, "hr_std": hr_std, "dhr": dhr},
            flags=flags,
        )
