from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from edf_extractor.constants import EPOCH_SEC, FEATURE_SPEC_VERSION


@dataclass(frozen=True)
class Thresholds:
    min_beats_per_epoch: int
    ecg_sqi_thresh: float
    ecg_rr_min_sec: float
    ecg_rr_max_sec: float
    hr_range_bpm: tuple[int, int]
    rr_range_bpm: tuple[int, int]
    min_breaths_per_epoch: int
    ecg_band_hz: tuple[float, float]
    resp_band_hz: tuple[float, float]
    move_minor_quantile: float
    move_large_quantile: float


@dataclass(frozen=True)
class ExtractConfig:
    feature_spec_version: str
    epoch_sec: int
    channel_aliases: dict[str, list[str]]
    thresholds: Thresholds

    def to_hash(self) -> str:
        payload = json.dumps(as_dict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path | None = None) -> ExtractConfig:
    config_path = Path(path) if path is not None else Path(__file__).with_name("defaults.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> ExtractConfig:
    thresholds_raw = raw["thresholds"]
    thresholds = Thresholds(
        min_beats_per_epoch=int(thresholds_raw["min_beats_per_epoch"]),
        ecg_sqi_thresh=float(thresholds_raw["ecg_sqi_thresh"]),
        ecg_rr_min_sec=float(thresholds_raw["ecg_rr_min_sec"]),
        ecg_rr_max_sec=float(thresholds_raw["ecg_rr_max_sec"]),
        hr_range_bpm=tuple(thresholds_raw["hr_range_bpm"]),
        rr_range_bpm=tuple(thresholds_raw["rr_range_bpm"]),
        min_breaths_per_epoch=int(thresholds_raw["min_breaths_per_epoch"]),
        ecg_band_hz=tuple(thresholds_raw["ecg_band_hz"]),
        resp_band_hz=tuple(thresholds_raw["resp_band_hz"]),
        move_minor_quantile=float(thresholds_raw["move_minor_quantile"]),
        move_large_quantile=float(thresholds_raw["move_large_quantile"]),
    )
    return ExtractConfig(
        feature_spec_version=str(raw.get("feature_spec_version", FEATURE_SPEC_VERSION)),
        epoch_sec=int(raw.get("epoch_sec", EPOCH_SEC)),
        channel_aliases=dict(raw.get("channel_aliases", {})),
        thresholds=thresholds,
    )


def as_dict(config: ExtractConfig) -> dict[str, Any]:
    return {
        "feature_spec_version": config.feature_spec_version,
        "epoch_sec": config.epoch_sec,
        "channel_aliases": config.channel_aliases,
        "thresholds": {
            "min_beats_per_epoch": config.thresholds.min_beats_per_epoch,
            "ecg_sqi_thresh": config.thresholds.ecg_sqi_thresh,
            "ecg_rr_min_sec": config.thresholds.ecg_rr_min_sec,
            "ecg_rr_max_sec": config.thresholds.ecg_rr_max_sec,
            "hr_range_bpm": list(config.thresholds.hr_range_bpm),
            "rr_range_bpm": list(config.thresholds.rr_range_bpm),
            "min_breaths_per_epoch": config.thresholds.min_breaths_per_epoch,
            "ecg_band_hz": list(config.thresholds.ecg_band_hz),
            "resp_band_hz": list(config.thresholds.resp_band_hz),
            "move_minor_quantile": config.thresholds.move_minor_quantile,
            "move_large_quantile": config.thresholds.move_large_quantile,
        },
    }
