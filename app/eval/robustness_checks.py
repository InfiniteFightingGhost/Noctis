from __future__ import annotations

from typing import Any

import numpy as np


def summarize_night_metric(nights: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = np.asarray([float(night[key]) for night in nights], dtype=np.float64)
    if vals.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "worst_decile_mean": 0.0,
            "worst_decile_ids": [],
        }
    p10 = float(np.percentile(vals, 10))
    k = max(int(np.ceil(0.1 * vals.size)), 1)
    idx = np.argsort(vals)[:k]
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p10": p10,
        "worst_decile_mean": float(np.mean(vals[idx])),
        "worst_decile_ids": [str(nights[i]["recording_id"]) for i in idx.tolist()],
    }


def clinical_mae(nights: list[dict[str, Any]], clinical_key: str) -> float:
    vals = np.asarray(
        [float(night["clinical_error"][clinical_key]["mae"]) for night in nights],
        dtype=np.float64,
    )
    return float(np.mean(vals)) if vals.size else 0.0


def clinical_ape(nights: list[dict[str, Any]], clinical_key: str) -> float:
    vals = np.asarray(
        [float(night["clinical_error"][clinical_key]["ape"]) for night in nights],
        dtype=np.float64,
    )
    return float(np.mean(vals)) if vals.size else 0.0


def min_class_recall(global_metrics: dict[str, Any]) -> float:
    recalls = [
        float(global_metrics["per_class"][class_name]["recall"])
        for class_name in sorted(global_metrics["per_class"].keys())
    ]
    return float(min(recalls)) if recalls else 0.0


def n1_recall(global_metrics: dict[str, Any]) -> float:
    if "N1" not in global_metrics["per_class"]:
        return 0.0
    return float(global_metrics["per_class"]["N1"]["recall"])


def class_recall(global_metrics: dict[str, Any], label: str) -> float:
    value = global_metrics["per_class"].get(label)
    if not isinstance(value, dict):
        return 0.0
    return float(value.get("recall", 0.0))


def run_gates(
    metrics: dict[str, Any],
    hard_thresholds: dict[str, float],
    soft_thresholds: dict[str, float],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    global_metrics = metrics["classification"]["global"]
    night_summary = metrics["night_summary"]
    domain_variances = metrics["domain"]["domain_variance"]
    clinical = metrics["clinical_agreement"]

    checks = [
        (
            "HARD_GLOBAL_MACRO_F1",
            float(global_metrics["macro_f1"]),
            float(hard_thresholds["macro_f1_min"]),
            "gte",
        ),
        (
            "HARD_GLOBAL_MCC",
            float(global_metrics["mcc"]),
            float(hard_thresholds["mcc_min"]),
            "gte",
        ),
        (
            "HARD_WORST_DECILE_MACRO_F1",
            float(night_summary["macro_f1"]["worst_decile_mean"]),
            float(hard_thresholds["worst_decile_macro_f1_min"]),
            "gte",
        ),
        (
            "HARD_DOMAIN_VARIANCE_MACRO_F1",
            float(domain_variances["macro_f1"]),
            float(hard_thresholds["domain_variance_macro_f1_max"]),
            "lte",
        ),
        (
            "HARD_REM_LATENCY_MAE",
            float(clinical["rem_latency_min"]["mae"]),
            float(hard_thresholds["rem_latency_mae_max"]),
            "lte",
        ),
        (
            "HARD_MIN_CLASS_RECALL",
            min_class_recall(global_metrics),
            float(hard_thresholds["min_class_recall_floor"]),
            "gte",
        ),
        (
            "HARD_REM_RECALL",
            class_recall(global_metrics, "REM"),
            float(hard_thresholds["rem_recall_floor"]),
            "gte",
        ),
        (
            "HARD_DEEP_RECALL",
            class_recall(global_metrics, "Deep"),
            float(hard_thresholds["deep_recall_floor"]),
            "gte",
        ),
        (
            "HARD_WORST_NIGHT_MACRO_F1",
            float(
                night_summary["macro_f1"].get(
                    "p10", night_summary["macro_f1"].get("worst_decile_mean", 0.0)
                )
            ),
            float(hard_thresholds["worst_night_macro_f1_floor"]),
            "gte",
        ),
        (
            "HARD_IMPOSSIBLE_TRANSITION_RATE",
            float(metrics["temporal_stability"]["impossible_transition_rate"]["mean"]),
            float(hard_thresholds["impossible_transition_rate_max"]),
            "lte",
        ),
        (
            "HARD_WAKE_DEEP_DIRECT_RATE",
            float(metrics["temporal_stability"]["wake_deep_direct_rate"]["mean"]),
            float(hard_thresholds["wake_deep_direct_rate_max"]),
            "lte",
        ),
    ]
    warn_checks = [
        (
            "SOFT_WORST_DECILE_MACRO_F1",
            float(night_summary["macro_f1"]["worst_decile_mean"]),
            float(soft_thresholds["worst_decile_macro_f1_warn"]),
            "gte",
        ),
    ]

    def evaluate(operator: str, value: float, threshold: float) -> bool:
        if operator == "gte":
            return value >= threshold
        return value <= threshold

    for reason_code, value, threshold, operator in checks:
        if not evaluate(operator, value, threshold):
            failures.append(
                {
                    "reason_code": reason_code,
                    "value": value,
                    "threshold": threshold,
                    "operator": operator,
                }
            )
    for reason_code, value, threshold, operator in warn_checks:
        if not evaluate(operator, value, threshold):
            warnings.append(
                {
                    "reason_code": reason_code,
                    "value": value,
                    "threshold": threshold,
                    "operator": operator,
                }
            )

    status = "PASS"
    if failures:
        status = "FAIL"
    elif warnings:
        status = "WARN"

    return {
        "status": status,
        "hard_failures": failures,
        "soft_warnings": warnings,
    }
