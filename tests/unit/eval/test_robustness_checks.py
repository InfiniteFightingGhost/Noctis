from __future__ import annotations

from app.eval.robustness_checks import run_gates


def test_run_gates_returns_reason_codes_for_hard_and_soft_checks() -> None:
    metrics = {
        "classification": {
            "global": {
                "macro_f1": 0.4,
                "mcc": 0.3,
                "per_class": {
                    "W": {"recall": 0.2},
                    "Light": {"recall": 0.4},
                    "Deep": {"recall": 0.2},
                    "REM": {"recall": 0.1},
                },
            }
        },
        "night_summary": {
            "macro_f1": {"worst_decile_mean": 0.35},
            "mcc": {"worst_decile_mean": 0.1},
            "kappa": {"worst_decile_mean": 0.1},
        },
        "domain": {"domain_variance": {"macro_f1": 0.3, "mcc": 0.0, "kappa": 0.0}},
        "clinical_agreement": {
            "rem_latency_min": {"mae": 35.0, "ape": 0.2},
        },
        "temporal_stability": {
            "impossible_transition_rate": {"mean": 0.2},
            "wake_deep_direct_rate": {"mean": 0.2},
        },
    }

    report = run_gates(
        metrics,
        hard_thresholds={
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
        },
        soft_thresholds={"worst_decile_macro_f1_warn": 0.45},
    )

    assert report["status"] == "FAIL"
    assert any(item["reason_code"] == "HARD_GLOBAL_MACRO_F1" for item in report["hard_failures"])
    assert any(
        item["reason_code"] == "SOFT_WORST_DECILE_MACRO_F1" for item in report["soft_warnings"]
    )
