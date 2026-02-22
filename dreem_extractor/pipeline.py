from __future__ import annotations

from pathlib import Path

import numpy as np

from dreem_extractor.channels.normalization import normalize_signal
from dreem_extractor.config import ExtractConfig
from dreem_extractor.constants import (
    AgreeBits,
    FEATURE_ORDER,
    FlagBits,
    INT8_UNKNOWN,
    UINT8_UNKNOWN,
)
from dreem_extractor.epoching.align import align_signal
from dreem_extractor.features.base import FeatureContext
from dreem_extractor.features.hr_ecg import ECGHRPlugin
from dreem_extractor.features.rr_edr import EDRPlugin
from dreem_extractor.features.rr_resp import RespRRPlugin
from dreem_extractor.features.unsupported import UnsupportedPlugin
from dreem_extractor.io.h5_reader import open_h5, read_dataset
from dreem_extractor.io.schema import build_manifest
from dreem_extractor.models import ExtractResult, SignalSeries
from dreem_extractor.qc.metrics import compute_qc


def extract_record(path: str | Path, config: ExtractConfig) -> ExtractResult:
    path = Path(path)
    warnings: list[str] = []
    with open_h5(path) as h5file:
        manifest = build_manifest(path, h5file, config)
        if manifest.hypnogram_path is None:
            raise ValueError(f"Missing hypnogram in {path}")
        hypnogram = read_dataset(h5file, manifest.hypnogram_path).astype(np.int8)
        n_epochs = int(hypnogram.shape[0])

        signals: dict[str, SignalSeries] = {}
        for logical, dataset_path in manifest.channel_map.items():
            data = normalize_signal(read_dataset(h5file, dataset_path))
            fs = manifest.fs_map.get(logical)
            if fs is None:
                continue
            segments, warn = align_signal(data, fs, n_epochs, config.epoch_sec)
            warnings.extend(warn)
            signals[logical] = SignalSeries(
                name=logical,
                data=data,
                fs=fs,
                segments=segments,
            )

    ctx = FeatureContext(config=config, n_epochs=n_epochs, signals=signals, hypnogram=hypnogram)

    features = _init_feature_arrays(n_epochs)
    flags = np.zeros(n_epochs, dtype=np.uint8)
    agree_flags = np.zeros(n_epochs, dtype=np.uint8)
    qc_metrics: dict[str, float] = {}

    for plugin in _select_plugins(signals):
        output = plugin.compute(ctx)
        for key, values in output.features.items():
            features[key] = values
        if output.flags is not None:
            flags |= output.flags
        if output.agree_flags is not None:
            agree_flags |= output.agree_flags
        if output.warnings:
            warnings.extend(output.warnings)
        if output.qc:
            qc_metrics.update(output.qc)

    if "ecg" in manifest.channel_map:
        flags |= 1 << FlagBits.ECG_PRESENT
    if "resp" in manifest.channel_map:
        flags |= 1 << FlagBits.RESP_PRESENT

    flags = _apply_stage_and_epoch_flags(flags, hypnogram, signals)
    agree_flags |= _compute_agree_flags(features, flags, ctx)
    features["flags"] = flags
    features["agree_flags"] = agree_flags

    feature_matrix = _stack_features(features)
    valid_mask = (hypnogram != -1) & ((flags & (1 << FlagBits.EPOCH_VALID)) > 0)

    timestamps = None
    if manifest.start_time is not None:
        timestamps = np.arange(n_epochs, dtype=np.int64) * config.epoch_sec

    metadata = {
        "record_id": manifest.record_id,
        "start_time": manifest.start_time,
        "channel_map": manifest.channel_map,
        "fs_map": manifest.fs_map,
        "warnings": warnings,
        "feature_spec_version": config.feature_spec_version,
        "extractor_version": "0.1.0",
        "config_hash": config.to_hash(),
        "feature_order": FEATURE_ORDER,
    }
    result = ExtractResult(
        record_id=manifest.record_id,
        hypnogram=hypnogram,
        features=feature_matrix,
        valid_mask=valid_mask,
        timestamps=timestamps,
        metadata=metadata,
        qc={},
        warnings=warnings,
    )
    result.qc = compute_qc(result, manifest, config=config, extra_metrics=qc_metrics)
    return result


def _select_plugins(signals: dict[str, SignalSeries]) -> list:
    plugins = [UnsupportedPlugin(), ECGHRPlugin()]
    if "resp" in signals:
        plugins.append(RespRRPlugin())
    else:
        plugins.append(EDRPlugin())
    return plugins


def _init_feature_arrays(n_epochs: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key in FEATURE_ORDER:
        if key in ("dhr", "drr"):
            arrays[key] = np.full(n_epochs, INT8_UNKNOWN, dtype=np.int8)
        elif key in ("flags", "agree_flags"):
            arrays[key] = np.zeros(n_epochs, dtype=np.uint8)
        else:
            arrays[key] = np.full(n_epochs, UINT8_UNKNOWN, dtype=np.uint8)
    return arrays


def _stack_features(features: dict[str, np.ndarray]) -> np.ndarray:
    columns = [features[key] for key in FEATURE_ORDER]
    return np.stack(columns, axis=1)


def _apply_stage_and_epoch_flags(
    flags: np.ndarray,
    hypnogram: np.ndarray,
    signals: dict[str, SignalSeries],
) -> np.ndarray:
    n_epochs = len(hypnogram)
    epoch_valid = np.zeros(n_epochs, dtype=bool)
    for series in signals.values():
        for idx, segment in enumerate(series.segments):
            if segment is not None:
                epoch_valid[idx] = True
    flags = flags.copy()
    flags[epoch_valid] |= 1 << FlagBits.EPOCH_VALID
    flags[hypnogram != -1] |= 1 << FlagBits.STAGE_SCORED
    return flags


def _compute_agree_flags(
    features: dict[str, np.ndarray],
    flags: np.ndarray,
    ctx: FeatureContext,
) -> np.ndarray:
    n_epochs = ctx.n_epochs
    agree = np.zeros(n_epochs, dtype=np.uint8)
    hr = features["hr_mean"].astype(float)
    rr = features["rr_mean"].astype(float)
    hr_valid = (flags & (1 << FlagBits.HR_VALID)) > 0
    rr_valid = (flags & (1 << FlagBits.RR_VALID)) > 0
    hr_min, hr_max = ctx.config.thresholds.hr_range_bpm
    rr_min, rr_max = ctx.config.thresholds.rr_range_bpm
    for i in range(n_epochs):
        if hr_valid[i] and hr_min <= hr[i] <= hr_max:
            agree[i] |= 1 << AgreeBits.HR_RANGE_OK
        if rr_valid[i] and rr_min <= rr[i] <= rr_max:
            agree[i] |= 1 << AgreeBits.RR_RANGE_OK
        if hr_valid[i] and rr_valid[i]:
            if rr[i] > 0 and hr[i] >= rr[i] and hr[i] <= 8 * rr[i]:
                agree[i] |= 1 << AgreeBits.HR_RR_PLAUSIBLE
    return agree
