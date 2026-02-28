from __future__ import annotations

from typing import Any

import numpy as np

from .calibration import calibration_metrics
from .domain_analysis import domain_variance, per_dataset_eval
from .metrics import CLASSES, evaluate_classification
from .night_evaluator import FORBIDDEN_DEFAULT, per_night_eval
from .robustness_checks import clinical_ape, clinical_mae, run_gates, summarize_night_metric

DEFAULT_HARD_THRESHOLDS = {
    "macro_f1_min": 0.75,
    "mcc_min": 0.70,
    "worst_decile_macro_f1_min": 0.50,
    "domain_variance_macro_f1_max": 0.15,
    "rem_latency_mae_max": 20.0,
    "min_class_recall_floor": 0.15,
    "rem_recall_floor": 0.65,
    "deep_recall_floor": 0.70,
    "worst_night_macro_f1_floor": 0.60,
    "impossible_transition_rate_max": 0.08,
    "wake_deep_direct_rate_max": 0.05,
}

DEFAULT_SOFT_THRESHOLDS = {
    "worst_decile_macro_f1_warn": 0.45,
}


def _clinical_summary(nights: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["tst_min", "se", "rem_pct", "n3_pct", "rem_latency_min", "waso_min"]
    out: dict[str, Any] = {}
    for key in keys:
        out[key] = {
            "mae": clinical_mae(nights, key),
            "ape": clinical_ape(nights, key),
        }
    return out


def evaluate_all(
    y_true,
    y_pred,
    proba,
    recording_id,
    dataset_id,
    *,
    class_names: list[str] | None = None,
    epoch_seconds: int = 30,
    calibration_bins: int = 15,
    forbidden_transitions: set[tuple[int, int]] = FORBIDDEN_DEFAULT,
    hard_thresholds: dict[str, float] | None = None,
    soft_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba = np.asarray(proba)

    num_classes = int(proba.shape[1])
    class_names = class_names or CLASSES[:num_classes]
    hard_thresholds = dict(DEFAULT_HARD_THRESHOLDS if hard_thresholds is None else hard_thresholds)
    soft_thresholds = dict(DEFAULT_SOFT_THRESHOLDS if soft_thresholds is None else soft_thresholds)

    cls_global = evaluate_classification(
        y_true,
        y_pred,
        num_classes=num_classes,
        class_names=class_names,
    )
    calibration = calibration_metrics(
        y_true,
        proba,
        n_bins=calibration_bins,
        class_names=class_names,
    )

    per_dataset = per_dataset_eval(
        y_true,
        y_pred,
        proba,
        dataset_id,
        class_names=class_names,
    )
    domain = {
        "per_dataset": per_dataset,
        "domain_variance": domain_variance(per_dataset),
    }

    night = per_night_eval(
        y_true,
        y_pred,
        proba,
        recording_id,
        dataset_id,
        class_names=class_names,
        epoch_seconds=epoch_seconds,
        calibration_bins=calibration_bins,
        forbidden_transitions=forbidden_transitions,
    )
    nights = night["nights"]

    night_summary = {
        "macro_f1": summarize_night_metric(nights, "macro_f1"),
        "mcc": summarize_night_metric(nights, "mcc"),
        "kappa": summarize_night_metric(nights, "kappa"),
    }
    temporal_stability = {
        "transition_matrix_error_raw": summarize_night_metric(
            nights, "transition_matrix_error_raw"
        ),
        "transition_matrix_error_norm": summarize_night_metric(
            nights, "transition_matrix_error_norm"
        ),
        "transition_entropy_diff": summarize_night_metric(nights, "transition_entropy_diff"),
        "hypnogram_edit_distance_raw": summarize_night_metric(
            nights, "hypnogram_edit_distance_raw"
        ),
        "hypnogram_edit_distance_norm": summarize_night_metric(
            nights, "hypnogram_edit_distance_norm"
        ),
        "fragmentation_error": summarize_night_metric(nights, "fragmentation_error"),
        "impossible_transition_rate": summarize_night_metric(nights, "impossible_transition_rate"),
        "wake_deep_direct_rate": summarize_night_metric(nights, "wake_deep_direct_rate"),
    }
    clinical_agreement = _clinical_summary(nights)

    scorecard = {
        "version": "2.0.0",
        "classification": {"global": cls_global},
        "calibration": calibration,
        "night": night,
        "night_summary": night_summary,
        "temporal_stability": temporal_stability,
        "clinical_agreement": clinical_agreement,
        "domain": domain,
    }
    robustness = run_gates(
        scorecard, hard_thresholds=hard_thresholds, soft_thresholds=soft_thresholds
    )
    scorecard["robustness"] = robustness
    scorecard["thresholds"] = {
        "hard": hard_thresholds,
        "soft": soft_thresholds,
    }
    return scorecard
