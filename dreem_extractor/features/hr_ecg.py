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
        threshold = np.median(filtered) + 3.0 * mad(filtered)
        peaks = detect_peaks(filtered, ecg.fs, min_distance_sec=0.25, height=threshold)
        if peaks.size < 2:
            return FeatureOutput(
                features={"hr_mean": hr_mean, "hr_std": hr_std, "dhr": dhr},
                flags=flags,
            )

        peak_times = peaks / ecg.fs
        rr_intervals_all = np.diff(peak_times)
        rr_times_all = peak_times[1:]
        if rr_intervals_all.size == 0:
            return FeatureOutput(
                features={"hr_mean": hr_mean, "hr_std": hr_std, "dhr": dhr},
                flags=flags,
            )
        rr_min_sec = ctx.config.thresholds.ecg_rr_min_sec
        rr_max_sec = ctx.config.thresholds.ecg_rr_max_sec
        valid_mask = (rr_intervals_all >= rr_min_sec) & (rr_intervals_all <= rr_max_sec)
        rr_intervals = rr_intervals_all[valid_mask]
        hr_values = 60.0 / rr_intervals
        hr_times = rr_times_all[valid_mask]

        min_hr, max_hr = ctx.config.thresholds.hr_range_bpm
        min_beats = ctx.config.thresholds.min_beats_per_epoch
        ecg_sqi_thresh = ctx.config.thresholds.ecg_sqi_thresh

        samples_per_epoch = int(round(ecg.fs * ctx.config.epoch_sec))
        for i in range(n_epochs):
            start_t = (i * samples_per_epoch) / ecg.fs
            end_t = ((i + 1) * samples_per_epoch) / ecg.fs
            rr_mask = (rr_times_all >= start_t) & (rr_times_all < end_t)
            total_rr = int(np.sum(rr_mask))
            valid_rr = int(np.sum(rr_mask & valid_mask))
            ecg_sqi = valid_rr / total_rr if total_rr else 0.0
            beat_count = int(np.sum((peak_times >= start_t) & (peak_times < end_t)))
            if beat_count < min_beats or ecg_sqi < ecg_sqi_thresh:
                continue
            mask = (hr_times >= start_t) & (hr_times < end_t)
            if not np.any(mask):
                continue
            values = hr_values[mask]
            mean_val = float(np.mean(values))
            if mean_val < min_hr or mean_val > max_hr:
                continue
            hr_mean[i] = int(round(min(255, max(0, mean_val))))
            hr_std[i] = int(round(min(255, max(0, float(np.std(values))))))
            flags[i] |= 1 << FlagBits.HR_VALID

        hr_values = _to_nan(hr_mean)
        hr_values = _post_process_hr(hr_values, ctx.config.thresholds)
        hr_jump_rate, hr_delta_p95 = _compute_delta_metrics(
            hr_values,
            ctx.config.thresholds.hr_jump_max_delta,
        )
        hr_values = _smooth_hr(
            hr_values,
            ctx.config.thresholds.hr_smooth_alpha,
            ctx.config.thresholds.hr_smooth_window,
        )
        _, hr_smooth_delta_p95 = _compute_delta_metrics(
            hr_values,
            ctx.config.thresholds.hr_jump_max_delta,
        )
        hr_mean = _from_float_to_uint8(hr_values)

        for i in range(1, n_epochs):
            if np.isnan(hr_values[i]) or np.isnan(hr_values[i - 1]):
                dhr[i] = INT8_UNKNOWN
            else:
                delta = int(round(hr_values[i] - hr_values[i - 1]))
                dhr[i] = int(max(-128, min(127, delta)))

        return FeatureOutput(
            features={"hr_mean": hr_mean, "hr_std": hr_std, "dhr": dhr},
            flags=flags,
            qc={
                "hr_jump_rate": hr_jump_rate,
                "hr_delta_p95": hr_delta_p95,
                "hr_smooth_delta_p95": hr_smooth_delta_p95,
            },
        )


def _to_nan(values: np.ndarray) -> np.ndarray:
    out = values.astype(float)
    out[values == UINT8_UNKNOWN] = np.nan
    return out


def _from_float_to_uint8(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, UINT8_UNKNOWN, dtype=np.uint8)
    mask = ~np.isnan(values)
    if np.any(mask):
        out[mask] = np.clip(np.round(values[mask]), 0, 255).astype(np.uint8)
    return out


def _post_process_hr(values: np.ndarray, thresholds) -> np.ndarray:
    out = _fill_short_nan_runs(values, thresholds.hr_gap_fill_max_epochs)
    out = _rolling_median(out, thresholds.hr_median_window)
    out = _cap_delta(out, thresholds.hr_jump_max_delta)
    return out


def _fill_short_nan_runs(values: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return values
    out = values.copy()
    last_value = np.nan
    run_start = -1
    for idx, value in enumerate(out):
        if np.isnan(value):
            if run_start < 0:
                run_start = idx
            continue
        if run_start >= 0:
            gap_len = idx - run_start
            if gap_len <= max_gap and not np.isnan(last_value):
                out[run_start:idx] = last_value
            run_start = -1
        last_value = value
    return out


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    out = values.copy()
    n = len(out)
    for i in range(n):
        if np.isnan(out[i]):
            continue
        start = max(0, i - window + 1)
        end = i + 1
        median_val = np.nanmedian(out[start:end])
        if not np.isnan(median_val):
            out[i] = median_val
    return out


def _cap_delta(values: np.ndarray, max_delta: float) -> np.ndarray:
    if max_delta <= 0:
        return values
    out = values.copy()
    prev = None
    for i in range(len(out)):
        if np.isnan(out[i]):
            prev = None
            continue
        if prev is None:
            prev = float(out[i])
            continue
        delta = float(out[i]) - prev
        if abs(delta) > max_delta:
            out[i] = prev + np.sign(delta) * max_delta
        prev = float(out[i])
    return out


def _smooth_hr(values: np.ndarray, alpha: float, window: int) -> np.ndarray:
    out = values.copy()
    if alpha > 0:
        prev = None
        for i in range(len(out)):
            if np.isnan(out[i]):
                prev = None
                continue
            if prev is None:
                prev = float(out[i])
                continue
            out[i] = alpha * float(out[i]) + (1.0 - alpha) * prev
            prev = float(out[i])
    if window <= 1:
        return out
    if window % 2 == 0:
        window += 1
    smoothed = out.copy()
    n = len(out)
    for i in range(n):
        if np.isnan(out[i]):
            continue
        start = max(0, i - window + 1)
        end = i + 1
        segment = out[start:end]
        if np.all(np.isnan(segment)):
            continue
        smoothed[i] = float(np.nanmean(segment))
    return smoothed


def _compute_delta_metrics(values: np.ndarray, max_delta: float) -> tuple[float, float]:
    deltas: list[float] = []
    jumps = 0
    prev = None
    for value in values:
        if np.isnan(value):
            prev = None
            continue
        if prev is None:
            prev = float(value)
            continue
        delta = abs(float(value) - prev)
        deltas.append(delta)
        if max_delta > 0 and delta > max_delta:
            jumps += 1
        prev = float(value)
    if not deltas:
        return 0.0, float("nan")
    jump_rate = jumps / len(deltas) if deltas else 0.0
    return jump_rate, float(np.percentile(np.array(deltas, dtype=float), 95))
