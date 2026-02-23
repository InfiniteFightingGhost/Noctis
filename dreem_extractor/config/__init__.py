from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dreem_extractor.constants import EPOCH_SEC, FEATURE_SPEC_VERSION


@dataclass(frozen=True)
class Thresholds:
    min_beats_per_epoch: int
    ecg_sqi_thresh: float
    ecg_rr_min_sec: float
    ecg_rr_max_sec: float
    hr_gap_fill_max_epochs: int
    hr_median_window: int
    hr_jump_max_delta: float
    hr_smooth_alpha: float
    hr_smooth_window: int
    hr_range_bpm: tuple[int, int]
    rr_range_bpm: tuple[int, int]
    rr_conf_min: float
    rr_agree_tol: float
    rr_smooth_alpha: float
    rr_gap_fill_max_epochs: int
    rr_quality_min: int
    edr_band_hz: tuple[float, float]
    edr_fs: float
    ecg_band_hz: tuple[float, float]
    resp_band_hz: tuple[float, float]
    min_breaths_per_epoch: int


@dataclass(frozen=True)
class ExtractConfig:
    feature_spec_version: str
    epoch_sec: int
    channel_patterns: dict[str, list[str]]
    fs_attr_keys: list[str]
    thresholds: Thresholds
    flags: dict[str, Any]

    def to_hash(self) -> str:
        payload = json.dumps(as_dict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path | None = None) -> ExtractConfig:
    if path is None:
        path = Path(__file__).with_name("defaults.yaml")
    else:
        path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> ExtractConfig:
    thresholds_raw = raw["thresholds"]
    thresholds = Thresholds(
        min_beats_per_epoch=int(thresholds_raw["min_beats_per_epoch"]),
        ecg_sqi_thresh=float(thresholds_raw["ecg_sqi_thresh"]),
        ecg_rr_min_sec=float(thresholds_raw["ecg_rr_min_sec"]),
        ecg_rr_max_sec=float(thresholds_raw["ecg_rr_max_sec"]),
        hr_gap_fill_max_epochs=int(thresholds_raw.get("hr_gap_fill_max_epochs", 0)),
        hr_median_window=int(thresholds_raw.get("hr_median_window", 1)),
        hr_jump_max_delta=float(thresholds_raw.get("hr_jump_max_delta", 0.0)),
        hr_smooth_alpha=float(thresholds_raw["hr_smooth_alpha"]),
        hr_smooth_window=int(thresholds_raw.get("hr_smooth_window", 0)),
        hr_range_bpm=tuple(thresholds_raw["hr_range_bpm"]),
        rr_range_bpm=tuple(thresholds_raw["rr_range_bpm"]),
        rr_conf_min=float(thresholds_raw["rr_conf_min"]),
        rr_agree_tol=float(thresholds_raw["rr_agree_tol"]),
        rr_smooth_alpha=float(thresholds_raw["rr_smooth_alpha"]),
        rr_gap_fill_max_epochs=int(thresholds_raw["rr_gap_fill_max_epochs"]),
        rr_quality_min=int(thresholds_raw["rr_quality_min"]),
        edr_band_hz=tuple(thresholds_raw["edr_band_hz"]),
        edr_fs=float(thresholds_raw["edr_fs"]),
        ecg_band_hz=tuple(thresholds_raw["ecg_band_hz"]),
        resp_band_hz=tuple(thresholds_raw["resp_band_hz"]),
        min_breaths_per_epoch=int(thresholds_raw["min_breaths_per_epoch"]),
    )
    return ExtractConfig(
        feature_spec_version=str(raw.get("feature_spec_version", FEATURE_SPEC_VERSION)),
        epoch_sec=int(raw.get("epoch_sec", EPOCH_SEC)),
        channel_patterns=dict(raw.get("channel_patterns", {})),
        fs_attr_keys=list(raw.get("fs_attr_keys", [])),
        thresholds=thresholds,
        flags=dict(raw.get("flags", {})),
    )


def as_dict(config: ExtractConfig) -> dict[str, Any]:
    return {
        "feature_spec_version": config.feature_spec_version,
        "epoch_sec": config.epoch_sec,
        "channel_patterns": config.channel_patterns,
        "fs_attr_keys": config.fs_attr_keys,
        "thresholds": {
            "min_beats_per_epoch": config.thresholds.min_beats_per_epoch,
            "ecg_sqi_thresh": config.thresholds.ecg_sqi_thresh,
            "ecg_rr_min_sec": config.thresholds.ecg_rr_min_sec,
            "ecg_rr_max_sec": config.thresholds.ecg_rr_max_sec,
            "hr_gap_fill_max_epochs": config.thresholds.hr_gap_fill_max_epochs,
            "hr_median_window": config.thresholds.hr_median_window,
            "hr_jump_max_delta": config.thresholds.hr_jump_max_delta,
            "hr_smooth_alpha": config.thresholds.hr_smooth_alpha,
            "hr_smooth_window": config.thresholds.hr_smooth_window,
            "hr_range_bpm": list(config.thresholds.hr_range_bpm),
            "rr_range_bpm": list(config.thresholds.rr_range_bpm),
            "rr_conf_min": config.thresholds.rr_conf_min,
            "rr_agree_tol": config.thresholds.rr_agree_tol,
            "rr_smooth_alpha": config.thresholds.rr_smooth_alpha,
            "rr_gap_fill_max_epochs": config.thresholds.rr_gap_fill_max_epochs,
            "rr_quality_min": config.thresholds.rr_quality_min,
            "edr_band_hz": list(config.thresholds.edr_band_hz),
            "edr_fs": config.thresholds.edr_fs,
            "ecg_band_hz": list(config.thresholds.ecg_band_hz),
            "resp_band_hz": list(config.thresholds.resp_band_hz),
            "min_breaths_per_epoch": config.thresholds.min_breaths_per_epoch,
        },
        "flags": config.flags,
    }
