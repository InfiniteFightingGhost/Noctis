from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

UINT8_UNKNOWN = 255
INT8_UNKNOWN = -1

FEATURE_KEYS: list[str] = [
    "in_bed_pct",
    "hr_mean",
    "hr_std",
    "dhr",
    "rr_mean",
    "rr_std",
    "drr",
    "large_move_pct",
    "minor_move_pct",
    "turnovers_delta",
    "apnea_delta",
    "flags",
    "vib_move_pct",
    "vib_resp_q",
    "agree_flags",
]


@dataclass(frozen=True)
class ExtractConfig:
    epoch_sec: int = 30
    drop_unknown_stages: bool = True
    export_windows: bool = False
    window_len: int = 21
    stride: int = 1
    label_mode: str = "center"
    fs_override: float | None = None
    minor_move_k: float = 3.0
    large_move_k: float = 6.0
    resp_band_low: float = 0.1
    resp_band_high: float = 0.5
    ecg_band_low: float = 0.5
    ecg_band_high: float = 40.0


def clamp_uint8(value: float) -> int:
    if value < 0:
        return 0
    if value > 255:
        return 255
    return int(round(value))


def clamp_int8(value: float) -> int:
    if value < -128:
        return -128
    if value > 127:
        return 127
    return int(round(value))


def ensure_feature_order(features: Iterable[str]) -> list[str]:
    ordered = list(features)
    if ordered != FEATURE_KEYS:
        raise ValueError("Feature order mismatch; must match FEATURE_KEYS")
    return ordered
