from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from extractor.config import FEATURE_KEYS, UINT8_UNKNOWN, ExtractConfig
from extractor.epoching import epoch_slices
from extractor.features.ecg import compute_ecg_features
from extractor.features.movement import compute_movement_features
from extractor.features.resp import compute_resp_features, compute_resp_quality
from extractor.h5io import discover_signals, load_hypnogram, open_h5
from extractor.hypnogram import map_hypnogram


@dataclass
class EpochRecord:
    recording_id: str
    epoch_index: int
    stage: int
    features: dict[str, int]


@dataclass
class ExtractResult:
    recording_id: str
    stages: np.ndarray
    stage_known: np.ndarray
    features: dict[str, np.ndarray]
    flags: np.ndarray
    records: list[EpochRecord]


def extract_recording(path: str | Path, config: ExtractConfig) -> ExtractResult:
    path = Path(path)
    recording_id = path.stem
    with open_h5(path) as h5file:
        hypnogram_raw = load_hypnogram(h5file)
        if hypnogram_raw is None:
            raise ValueError(f"Missing hypnogram in {path}")
        stages, stage_known = map_hypnogram(hypnogram_raw)
        n_epochs = len(stages)

        signals = discover_signals(h5file, fs_override=config.fs_override)

    ecg_signal = signals.get("ecg")
    resp_signal = signals.get("resp")
    emg_signal = signals.get("emg")
    vib_signal = signals.get("vib")
    bed_signal = signals.get("bed")
    apnea_signal = signals.get("apnea")
    movement_signal = signals.get("move")

    hr_mean, hr_std, dhr, ecg_valid = compute_ecg_features(
        ecg_signal,
        n_epochs,
        config,
    )

    rr_mean, rr_std, drr, resp_valid = compute_resp_features(
        resp_signal,
        ecg_signal,
        n_epochs,
        config,
    )

    if movement_signal is None:
        movement_signal = emg_signal

    large_move_pct, minor_move_pct, turnovers_delta, move_valid = (
        compute_movement_features(
            movement_signal,
            n_epochs,
            config,
        )
    )

    vib_move_pct, vib_resp_q, vib_rr_mean, vib_valid = compute_vib_features(
        vib_signal,
        n_epochs,
        config,
    )

    in_bed_pct = compute_in_bed(bed_signal, n_epochs, config)
    apnea_delta = compute_apnea_delta(apnea_signal, n_epochs)

    agree_flags = compute_agree_flags(rr_mean, vib_rr_mean)

    flags = build_flags(
        stage_known=stage_known,
        ecg_valid=ecg_valid,
        resp_valid=resp_valid,
        move_valid=move_valid,
        vib_valid=vib_valid,
        in_bed_known=in_bed_pct != UINT8_UNKNOWN,
    )

    feature_arrays: dict[str, np.ndarray] = {
        "in_bed_pct": in_bed_pct,
        "hr_mean": hr_mean,
        "hr_std": hr_std,
        "dhr": dhr,
        "rr_mean": rr_mean,
        "rr_std": rr_std,
        "drr": drr,
        "large_move_pct": large_move_pct,
        "minor_move_pct": minor_move_pct,
        "turnovers_delta": turnovers_delta,
        "apnea_delta": apnea_delta,
        "flags": flags,
        "vib_move_pct": vib_move_pct,
        "vib_resp_q": vib_resp_q,
        "agree_flags": agree_flags,
    }

    for key in FEATURE_KEYS:
        if key not in feature_arrays:
            raise ValueError(f"Missing feature key {key}")
        if len(feature_arrays[key]) != n_epochs:
            raise ValueError(f"Feature {key} length mismatch")

    records = build_records(recording_id, stages, feature_arrays)

    return ExtractResult(
        recording_id=recording_id,
        stages=stages,
        stage_known=stage_known,
        features=feature_arrays,
        flags=flags,
        records=records,
    )


def compute_in_bed(
    signal: dict[str, Any] | None, n_epochs: int, config: ExtractConfig
) -> np.ndarray:
    if signal is None:
        return np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    data = signal["data"]
    fs = signal["fs"]
    if fs is None:
        return np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    slices = epoch_slices(len(data), fs, config.epoch_sec, n_epochs)
    values = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    for i, (start, end) in enumerate(slices):
        if end > len(data):
            continue
        segment = data[start:end]
        if segment.size == 0:
            continue
        threshold = 0.5 if np.issubdtype(segment.dtype, np.floating) else 0
        pct = 100.0 * float(np.mean(segment > threshold))
        values[i] = int(round(min(255, max(0, pct))))
    return values


def compute_apnea_delta(signal: dict[str, Any] | None, n_epochs: int) -> np.ndarray:
    if signal is None:
        return np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    data = signal["data"]
    if len(data) != n_epochs:
        return np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    data = np.asarray(data)
    deltas = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    if data.size == 0:
        return deltas
    prev = None
    for i in range(n_epochs):
        if prev is None:
            deltas[i] = UINT8_UNKNOWN
            prev = data[i]
            continue
        delta = data[i] - prev
        deltas[i] = int(round(min(255, max(0, delta))))
        prev = data[i]
    return deltas


def compute_vib_features(
    signal: dict[str, Any] | None,
    n_epochs: int,
    config: ExtractConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if signal is None:
        unknown_u8 = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
        unknown_rr = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
        return unknown_u8, unknown_u8, unknown_rr, np.zeros(n_epochs, dtype=bool)
    _, vib_move_pct, _, move_valid = compute_movement_features(signal, n_epochs, config)
    rr_mean, _, _, resp_valid = compute_resp_features(signal, None, n_epochs, config)
    resp_q = compute_resp_quality(signal, n_epochs, config)
    vib_valid = move_valid | resp_valid
    return vib_move_pct, resp_q, rr_mean, vib_valid


def compute_agree_flags(rr_mean: np.ndarray, vib_rr_mean: np.ndarray) -> np.ndarray:
    flags = np.full(len(rr_mean), UINT8_UNKNOWN, dtype=np.uint8)
    for i in range(len(rr_mean)):
        if rr_mean[i] == UINT8_UNKNOWN or vib_rr_mean[i] == UINT8_UNKNOWN:
            continue
        diff = abs(int(rr_mean[i]) - int(vib_rr_mean[i]))
        agree = 1 if diff <= 2 else 0
        flags[i] = agree
    return flags


def build_flags(
    *,
    stage_known: np.ndarray,
    ecg_valid: np.ndarray,
    resp_valid: np.ndarray,
    move_valid: np.ndarray,
    vib_valid: np.ndarray,
    in_bed_known: np.ndarray,
) -> np.ndarray:
    n_epochs = len(stage_known)
    flags = np.zeros(n_epochs, dtype=np.uint8)
    for i in range(n_epochs):
        epoch_valid = bool(
            ecg_valid[i]
            or resp_valid[i]
            or move_valid[i]
            or vib_valid[i]
            or in_bed_known[i]
        )
        if epoch_valid:
            flags[i] |= 1 << 0
        if ecg_valid[i]:
            flags[i] |= 1 << 1
        if resp_valid[i]:
            flags[i] |= 1 << 2
        if move_valid[i]:
            flags[i] |= 1 << 3
        if vib_valid[i]:
            flags[i] |= 1 << 4
        if stage_known[i]:
            flags[i] |= 1 << 5
    return flags


def build_records(
    recording_id: str,
    stages: np.ndarray,
    features: dict[str, np.ndarray],
) -> list[EpochRecord]:
    records: list[EpochRecord] = []
    for idx, stage in enumerate(stages):
        feature_dict: dict[str, int] = {}
        for key in FEATURE_KEYS:
            value = features[key][idx]
            feature_dict[key] = int(value)
        records.append(
            EpochRecord(
                recording_id=recording_id,
                epoch_index=idx,
                stage=int(stage),
                features=feature_dict,
            )
        )
    return records
