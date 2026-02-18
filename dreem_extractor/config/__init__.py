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
    hr_range_bpm: tuple[int, int]
    rr_range_bpm: tuple[int, int]
    rr_quality_min: int
    edr_band_hz: tuple[float, float]
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
    thresholds = Thresholds(
        min_beats_per_epoch=int(raw["thresholds"]["min_beats_per_epoch"]),
        hr_range_bpm=tuple(raw["thresholds"]["hr_range_bpm"]),
        rr_range_bpm=tuple(raw["thresholds"]["rr_range_bpm"]),
        rr_quality_min=int(raw["thresholds"]["rr_quality_min"]),
        edr_band_hz=tuple(raw["thresholds"]["edr_band_hz"]),
        ecg_band_hz=tuple(raw["thresholds"]["ecg_band_hz"]),
        resp_band_hz=tuple(raw["thresholds"]["resp_band_hz"]),
        min_breaths_per_epoch=int(raw["thresholds"]["min_breaths_per_epoch"]),
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
            "hr_range_bpm": list(config.thresholds.hr_range_bpm),
            "rr_range_bpm": list(config.thresholds.rr_range_bpm),
            "rr_quality_min": config.thresholds.rr_quality_min,
            "edr_band_hz": list(config.thresholds.edr_band_hz),
            "ecg_band_hz": list(config.thresholds.ecg_band_hz),
            "resp_band_hz": list(config.thresholds.resp_band_hz),
            "min_breaths_per_epoch": config.thresholds.min_breaths_per_epoch,
        },
        "flags": config.flags,
    }
