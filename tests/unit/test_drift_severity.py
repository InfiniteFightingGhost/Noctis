from __future__ import annotations

from app.drift import router as drift_router


def test_overall_severity_escalates_on_alert() -> None:
    metrics = [{"severity": "LOW"}]
    feature_drift = []
    severity = drift_router._overall_severity(metrics, feature_drift, alert_count=1)
    assert severity == "MEDIUM"


def test_severity_value_mapping() -> None:
    assert drift_router._severity_value("LOW") == 1
    assert drift_router._severity_value("MEDIUM") == 2
    assert drift_router._severity_value("HIGH") == 3
