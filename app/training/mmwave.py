from __future__ import annotations

from typing import Any

import numpy as np


EPSILON = 1e-6
FOUR_CLASS_LABELS = ["W", "Light", "Deep", "REM"]
ALLOWED_BASE_FEATURES = [
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
ENGINEERED_FEATURE_NAMES = [
    "hr_cv",
    "rr_cv",
    "instability",
    "move_total",
    "move_ratio",
    "stillness",
    "hr_rr_ratio",
    "var_ratio",
    "resp_instability",
    "vib_resp_instability",
    "low_agreement",
]


def remap_stage_label(label: Any) -> str | None:
    if label is None:
        return None
    value = str(label).strip().upper()
    if value in {"W", "WAKE"}:
        return "W"
    if value in {"N1", "N2", "LIGHT"}:
        return "Light"
    if value in {"N3", "DEEP"}:
        return "Deep"
    if value == "REM":
        return "REM"
    return None


def validate_base_feature_schema(feature_names: list[str]) -> None:
    provided = set(feature_names)
    allowed = set(ALLOWED_BASE_FEATURES)
    if not provided.issubset(allowed):
        extra = sorted(provided - allowed)
        raise ValueError(f"Unsupported base features for mmWave pipeline: {extra}")
    required = {
        "hr_mean",
        "hr_std",
        "dhr",
        "rr_mean",
        "rr_std",
        "drr",
        "large_move_pct",
        "minor_move_pct",
        "vib_move_pct",
        "vib_resp_q",
        "flags",
        "agree_flags",
    }
    missing = sorted(required - provided)
    if missing:
        raise ValueError(f"Missing required base features for engineered pipeline: {missing}")


def low_agreement_threshold_from_train(
    X: np.ndarray,
    *,
    feature_names: list[str],
    train_indices: np.ndarray,
) -> float:
    agree = _feature(X[train_indices], feature_names, "agree_flags").reshape(-1)
    agree = agree[np.isfinite(agree)]
    if agree.size == 0:
        return 0.5
    return float(np.quantile(agree, 0.15))


def engineer_mmwave_features(
    X: np.ndarray,
    *,
    feature_names: list[str],
    low_agreement_threshold: float,
    eps: float = EPSILON,
) -> tuple[np.ndarray, list[str], dict[str, str]]:
    validate_base_feature_schema(feature_names)
    hr_mean = _feature(X, feature_names, "hr_mean")
    hr_std = _feature(X, feature_names, "hr_std")
    dhr = _feature(X, feature_names, "dhr")
    rr_mean = _feature(X, feature_names, "rr_mean")
    rr_std = _feature(X, feature_names, "rr_std")
    drr = _feature(X, feature_names, "drr")
    large_move = _feature(X, feature_names, "large_move_pct")
    minor_move = _feature(X, feature_names, "minor_move_pct")
    vib_move = _feature(X, feature_names, "vib_move_pct")
    vib_resp_q = _feature(X, feature_names, "vib_resp_q")
    agree_flags = _feature(X, feature_names, "agree_flags")

    move_total = large_move + minor_move + vib_move
    engineered = {
        "hr_cv": hr_std / (np.abs(hr_mean) + eps),
        "rr_cv": rr_std / (np.abs(rr_mean) + eps),
        "instability": hr_std + rr_std + np.abs(dhr) + np.abs(drr),
        "move_total": move_total,
        "move_ratio": large_move / (minor_move + eps),
        "stillness": 1.0 - np.clip(move_total, 0.0, 1.0),
        "hr_rr_ratio": hr_mean / (rr_mean + eps),
        "var_ratio": hr_std / (rr_std + eps),
        "resp_instability": _feature(X, feature_names, "apnea_delta") * rr_std,
        "vib_resp_instability": vib_resp_q * rr_std,
        "low_agreement": (agree_flags < float(low_agreement_threshold)).astype(np.float32),
    }
    extra = np.stack([engineered[name] for name in ENGINEERED_FEATURE_NAMES], axis=-1)
    final = np.concatenate([X, extra.astype(np.float32)], axis=-1).astype(np.float32)
    formulas = {
        "hr_cv": "hr_std / (abs(hr_mean) + 1e-6)",
        "rr_cv": "rr_std / (abs(rr_mean) + 1e-6)",
        "instability": "hr_std + rr_std + abs(dhr) + abs(drr)",
        "move_total": "large_move_pct + minor_move_pct + vib_move_pct",
        "move_ratio": "large_move_pct / (minor_move_pct + 1e-6)",
        "stillness": "1 - clip(move_total, 0, 1)",
        "hr_rr_ratio": "hr_mean / (rr_mean + 1e-6)",
        "var_ratio": "hr_std / (rr_std + 1e-6)",
        "resp_instability": "apnea_delta * rr_std",
        "vib_resp_instability": "vib_resp_q * rr_std",
        "low_agreement": "1 if agree_flags < train_quantile_0.15(agree_flags) else 0",
    }
    return final, feature_names + ENGINEERED_FEATURE_NAMES, formulas


def _feature(X: np.ndarray, feature_names: list[str], name: str) -> np.ndarray:
    index = feature_names.index(name)
    values = np.asarray(X[:, :, index], dtype=np.float32)
    values = np.where(np.isfinite(values), values, 0.0)
    return values
