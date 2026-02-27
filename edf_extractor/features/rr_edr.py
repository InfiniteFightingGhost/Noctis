from __future__ import annotations

import numpy as np

from edf_extractor.constants import FlagBits, INT8_UNKNOWN, UINT8_UNKNOWN
from edf_extractor.features.base import FeatureContext, FeatureOutput, FeaturePlugin
from edf_extractor.features.utils import bandpass_filter, detect_peaks, mad


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
        edr_fs = ctx.config.thresholds.edr_fs
        duration = len(ecg.data) / ecg.fs
        times = np.arange(0, duration, 1.0 / edr_fs)
        edr = np.interp(times, peak_times, peak_amp, left=peak_amp[0], right=peak_amp[-1])
        edr = bandpass_filter(
            edr,
            edr_fs,
            ctx.config.thresholds.edr_band_hz[0],
            ctx.config.thresholds.edr_band_hz[1],
        )

        rr_psd_series = np.zeros(n_epochs, dtype=np.float32)
        rr_ac_series = np.zeros(n_epochs, dtype=np.float32)
        rr_peak_series = np.zeros(n_epochs, dtype=np.float32)
        band_ratio_series = np.zeros(n_epochs, dtype=np.float32)
        peak_prom_series = np.zeros(n_epochs, dtype=np.float32)
        ac_peak_series = np.zeros(n_epochs, dtype=np.float32)
        rr_std_raw = np.zeros(n_epochs, dtype=np.float32)

        samples_per_epoch = int(round(edr_fs * ctx.config.epoch_sec))
        for i in range(n_epochs):
            start = i * samples_per_epoch
            end = start + samples_per_epoch
            if end > edr.size:
                continue
            segment = edr[start:end]
            rr_psd, band_ratio, peak_prom = _rr_from_psd(
                segment, edr_fs, ctx.config.thresholds.edr_band_hz
            )
            rr_ac, ac_peak = _rr_from_autocorr(segment, edr_fs, ctx.config.thresholds.edr_band_hz)
            rr_peak = _rr_from_peaks(segment, edr_fs, ctx.config.thresholds.edr_band_hz)

            rr_psd_series[i] = rr_psd
            rr_ac_series[i] = rr_ac
            rr_peak_series[i] = rr_peak
            band_ratio_series[i] = band_ratio
            peak_prom_series[i] = peak_prom
            ac_peak_series[i] = ac_peak

            win = int(edr_fs * 10)
            if win > 0 and segment.size >= win * 2:
                sub_rr = []
                for k in range(segment.size // win):
                    s0 = k * win
                    s1 = s0 + win
                    rr_sub, _, _ = _rr_from_psd(
                        segment[s0:s1], edr_fs, ctx.config.thresholds.edr_band_hz
                    )
                    if rr_sub > 0:
                        sub_rr.append(rr_sub)
                if len(sub_rr) >= 2:
                    rr_std_raw[i] = float(np.std(sub_rr))

        rr_fused = np.zeros(n_epochs, dtype=np.float32)
        rr_conf = np.zeros(n_epochs, dtype=np.float32)
        rr_conf_min = ctx.config.thresholds.rr_conf_min
        rr_agree_tol = ctx.config.thresholds.rr_agree_tol
        rr_smooth_alpha = ctx.config.thresholds.rr_smooth_alpha
        min_rr, max_rr = ctx.config.thresholds.rr_range_bpm

        for i in range(n_epochs):
            ests = []
            weights = []
            band_quality = _quality_score(band_ratio_series[i], scale=0.4)
            peak_quality = _quality_score(peak_prom_series[i], scale=5.0)
            ac_quality = _quality_score(ac_peak_series[i], scale=0.8)
            if rr_psd_series[i] > 0:
                ests.append(rr_psd_series[i])
                weights.append(max(band_quality, peak_quality))
            if rr_ac_series[i] > 0:
                ests.append(rr_ac_series[i])
                weights.append(ac_quality if ac_quality > 0 else 0.5)
            if rr_peak_series[i] > 0:
                ests.append(rr_peak_series[i])
                weights.append(peak_quality if peak_quality > 0 else 0.5)

            if not ests:
                continue

            ests_arr = np.array(ests, dtype=float)
            weights_arr = np.array(weights, dtype=float)
            rr_fused[i] = (
                float(np.average(ests_arr, weights=weights_arr))
                if np.sum(weights_arr) > 0
                else float(np.median(ests_arr))
            )
            if len(ests_arr) >= 2:
                med = float(np.median(ests_arr))
                mad_val = float(np.median(np.abs(ests_arr - med)))
                tol = max(rr_agree_tol, 0.1 * med)
                agree = max(0.0, 1.0 - (mad_val / tol))
            else:
                agree = 0.5

            edr_sqi = float(0.4 * band_quality + 0.4 * peak_quality + 0.2 * ac_quality)
            rr_conf[i] = float(np.clip(0.6 * edr_sqi + 0.4 * agree, 0.0, 1.0))

        prev_rr = 0.0
        for i in range(n_epochs):
            rr_val = float(rr_fused[i])
            if rr_val <= 0:
                prev_rr = 0.0
                continue
            if rr_conf[i] < rr_conf_min:
                if rr_conf[i] > 0 and prev_rr > 0:
                    rr_val = rr_smooth_alpha * rr_val + (1.0 - rr_smooth_alpha) * prev_rr
                else:
                    prev_rr = 0.0
                    continue
            if rr_val < min_rr or rr_val > max_rr:
                prev_rr = 0.0
                continue
            rr_mean[i] = int(round(min(255, max(0, rr_val))))
            rr_std[i] = int(round(min(255, max(0, rr_std_raw[i]))))
            if prev_rr > 0:
                drr[i] = int(max(-128, min(127, rr_val - prev_rr)))
            prev_rr = rr_val
            flags[i] |= 1 << FlagBits.RR_VALID
            vib_resp_q[i] = int(round(min(255.0, max(0.0, rr_conf[i] * 255.0))))

        return FeatureOutput(
            features={"rr_mean": rr_mean, "rr_std": rr_std, "drr": drr, "vib_resp_q": vib_resp_q},
            flags=flags,
        )


def _rr_from_psd(
    edr_seg: np.ndarray, fs: float, band: tuple[float, float]
) -> tuple[float, float, float]:
    edr_seg = np.asarray(edr_seg, dtype=np.float32)
    if edr_seg.size < max(8, int(fs * 2)):
        return 0.0, 0.0, 0.0
    edr_seg = edr_seg - np.mean(edr_seg)
    freqs = np.fft.rfftfreq(edr_seg.size, d=1.0 / fs)
    pxx = np.abs(np.fft.rfft(edr_seg)) ** 2
    if freqs.size == 0:
        return 0.0, 0.0, 0.0
    inband = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(inband):
        return 0.0, 0.0, 0.0
    p_in = pxx[inband]
    total_power = float(np.sum(pxx))
    band_power = float(np.sum(p_in))
    if band_power <= 0:
        return 0.0, 0.0, 0.0
    f_in = freqs[inband]
    idx = int(np.argmax(p_in))
    peak_f = float(f_in[idx])
    peak_prom = float(p_in[idx] / (np.median(p_in) + 1e-9))
    band_ratio = float(band_power / (total_power + 1e-9))
    rr_bpm = peak_f * 60.0
    return rr_bpm, band_ratio, peak_prom


def _rr_from_autocorr(
    edr_seg: np.ndarray, fs: float, band: tuple[float, float]
) -> tuple[float, float]:
    edr_seg = np.asarray(edr_seg, dtype=np.float32)
    if edr_seg.size < 8:
        return 0.0, 0.0
    x = edr_seg - np.mean(edr_seg)
    corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    if corr.size < 2 or corr[0] == 0:
        return 0.0, 0.0
    corr = corr / corr[0]
    min_lag = max(1, int(fs / band[1]))
    max_lag = min(corr.size - 1, int(fs / band[0]))
    if max_lag <= min_lag:
        return 0.0, 0.0
    segment = corr[min_lag : max_lag + 1]
    idx = int(np.argmax(segment))
    lag = min_lag + idx
    peak_val = float(segment[idx])
    rr_bpm = 60.0 / (lag / fs)
    return rr_bpm, peak_val


def _rr_from_peaks(edr_seg: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    edr_seg = np.asarray(edr_seg, dtype=np.float32)
    if edr_seg.size < 8:
        return 0.0
    min_dist = max(1, int(fs / band[1]))
    peaks = detect_peaks(edr_seg, fs, min_distance_sec=min_dist / fs)
    if peaks.size < 2:
        return 0.0
    intervals = np.diff(peaks) / fs
    min_period = 1.0 / band[1]
    max_period = 1.0 / band[0]
    valid = intervals[(intervals >= min_period) & (intervals <= max_period)]
    if valid.size == 0:
        return 0.0
    return float(60.0 / np.median(valid))


def _quality_score(value: float, scale: float) -> float:
    if not np.isfinite(value) or value <= 0 or scale <= 0:
        return 0.0
    return float(np.clip(value / scale, 0.0, 1.0))
