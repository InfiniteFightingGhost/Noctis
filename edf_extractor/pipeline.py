from __future__ import annotations

from pathlib import Path
from dataclasses import asdict

import numpy as np

from edf_extractor.config import ExtractConfig, as_dict as config_as_dict
from edf_extractor.constants import (
    AgreeBits,
    EXTRACTOR_VERSION,
    FEATURE_ORDER,
    FlagBits,
    INT8_UNKNOWN,
    UINT8_UNKNOWN,
)
from edf_extractor.features.base import FeatureContext
from edf_extractor.features.hr_ecg import ECGHRPlugin
from edf_extractor.features.rr_edr import EDRPlugin
from edf_extractor.features.rr_resp import RespRRPlugin
from edf_extractor.features.unsupported import UnsupportedPlugin
from edf_extractor.io.edf_reader import read_edf_record
from edf_extractor.io.hypnogram import merge_hypnogram_tracks, read_hypnogram
from edf_extractor.io.schema import build_manifest
from edf_extractor.models import ExtractResult, SignalSeries
from edf_extractor.qc.metrics import compute_qc
from extractor_hardened.alignment import align_signal_deterministic
from extractor_hardened.contracts import load_contracts
from extractor_hardened.errors import ExtractionError
from extractor_hardened.qc import run_qc


def extract_record(
    edf_path: str | Path,
    cap_path: str | Path,
    config: ExtractConfig,
) -> ExtractResult:
    contracts = load_contracts()
    alignment_mode = str(contracts.alignment_policy.get("mode", "reconcile"))
    is_rec_input = Path(edf_path).suffix.lower() == ".rec"
    warnings: list[str] = []
    edf_record = read_edf_record(edf_path)
    hypnogram, cap_start_time, hyp_warnings = read_hypnogram(cap_path, config.epoch_sec)
    warnings.extend(hyp_warnings)
    hypnogram, annotation_warnings = merge_hypnogram_tracks(
        hypnogram,
        config.epoch_sec,
        edf_record.annotations,
    )
    warnings.extend(annotation_warnings)
    epoch_offset, offset_warnings = _resolve_epoch_offset(
        edf_start_time=edf_record.start_time,
        cap_start_time=cap_start_time,
        epoch_sec=config.epoch_sec,
    )
    warnings.extend(offset_warnings)

    start_time = edf_record.start_time or cap_start_time
    manifest = build_manifest(
        edf_path=edf_path,
        signals=edf_record.signals,
        config=config,
        cap_path=cap_path,
        start_time=start_time,
    )

    n_epochs = int(hypnogram.shape[0])
    signals: dict[str, SignalSeries] = {}
    alignment_decisions: dict[str, dict[str, object]] = {}
    for logical, idx in manifest.channel_index_map.items():
        signal = edf_record.signals[idx]
        aligned = align_signal_deterministic(
            signal.data,
            signal.fs,
            n_epochs,
            config.epoch_sec,
            mode=alignment_mode,
            epoch_offset=epoch_offset,
        )
        if aligned.decision.status != "exact":
            warnings.append("signal_length_reconciled")
        alignment_decisions[logical] = {
            "status": aligned.decision.status,
            "reason": aligned.decision.reason,
            "expected_samples": aligned.decision.expected_samples,
            "available_samples": aligned.decision.available_samples,
            "epoch_offset": epoch_offset,
        }
        signals[logical] = SignalSeries(
            name=logical,
            data=signal.data,
            fs=signal.fs,
            segments=aligned.segments,
        )

    required_channels = set(contracts.qc_policy.get("required_channels", []))
    strict_required_presence = bool(contracts.qc_policy.get("strict_required_presence", False))
    if is_rec_input and "ecg" in required_channels and "ecg" not in signals:
        required_channels.remove("ecg")
        warnings.append("required_ecg_not_found_for_rec")
    qc_policy = dict(contracts.qc_policy)
    qc_policy["required_channels"] = sorted(required_channels)
    missing_required = sorted([name for name in required_channels if name not in signals])
    if missing_required and strict_required_presence:
        raise ExtractionError(
            code="E_QC_FAIL",
            message="Missing required channels",
            details={"missing_required_channels": missing_required},
        )
    if missing_required:
        warnings.append("missing_required_channels")

    channel_qc, qc_summary = run_qc(signals, manifest.fs_map, qc_policy)
    optional_failed = set(qc_summary.get("optional_failed_channels", []))
    if optional_failed:
        signals = {name: value for name, value in signals.items() if name not in optional_failed}

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

    flags = _apply_stage_and_epoch_flags(flags, hypnogram, signals)
    agree_flags |= _compute_agree_flags(features, flags, ctx)
    features["flags"] = flags
    features["agree_flags"] = agree_flags

    feature_matrix = _stack_features(features)
    valid_mask = (hypnogram != -1) & ((flags & (1 << FlagBits.EPOCH_VALID)) > 0)
    timestamps = np.arange(n_epochs, dtype=np.int64) * config.epoch_sec if start_time else None

    metadata = {
        "record_id": manifest.record_id,
        "source_path": str(edf_path),
        "hypnogram_ref": manifest.hypnogram_ref,
        "start_time": manifest.start_time,
        "channel_map": manifest.channel_map,
        "channel_index_map": manifest.channel_index_map,
        "fs_map": manifest.fs_map,
        "cap_start_time": cap_start_time,
        "epoch_offset": epoch_offset,
        "warnings": warnings,
        "feature_spec_version": config.feature_spec_version,
        "extractor_version": EXTRACTOR_VERSION,
        "config_hash": config.to_hash(),
        "config_payload": config_as_dict(config),
        "feature_order": FEATURE_ORDER,
        "alignment_decisions": alignment_decisions,
        "qc_summary": qc_summary,
        "channel_qc": {name: asdict(report) for name, report in channel_qc.items()},
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


def _resolve_epoch_offset(
    edf_start_time: str | None,
    cap_start_time: str | None,
    epoch_sec: int,
) -> tuple[int, list[str]]:
    warnings: list[str] = []
    if epoch_sec <= 0:
        return 0, warnings
    edf_clock = _parse_clock_seconds(edf_start_time)
    cap_clock = _parse_clock_seconds(cap_start_time)
    if edf_clock is None or cap_clock is None:
        return 0, warnings

    delta_sec = cap_clock - edf_clock
    if delta_sec > 12 * 3600:
        delta_sec -= 24 * 3600
    elif delta_sec < -12 * 3600:
        delta_sec += 24 * 3600

    epoch_offset = int(round(delta_sec / epoch_sec))
    if epoch_offset != 0:
        warnings.append("cap_edf_start_offset_applied")
    if abs(delta_sec - epoch_offset * epoch_sec) > 1:
        warnings.append("cap_edf_start_offset_non_multiple_epoch")
    return epoch_offset, warnings


def _parse_clock_seconds(value: str | None) -> int | None:
    if value is None:
        return None
    parts = value.split("T")[-1].replace(".", ":").split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
    except ValueError:
        return None
    if hours < 0 or minutes < 0 or seconds < 0:
        return None
    if minutes >= 60 or seconds >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds
