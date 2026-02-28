from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.promotion.service import _assert_metrics_thresholds


def _settings():
    return SimpleNamespace(
        promotion_block_if_missing_metrics=True,
        promotion_min_accuracy=0.6,
        promotion_min_macro_f1=0.5,
    )


def test_metrics_thresholds_accept_nested_scorecard(monkeypatch) -> None:
    monkeypatch.setattr("app.promotion.service.get_settings", _settings)
    model = SimpleNamespace(
        metrics={
            "classification": {
                "global": {
                    "balanced_accuracy": 0.7,
                    "macro_f1": 0.6,
                }
            }
        }
    )
    _assert_metrics_thresholds(model)


def test_metrics_thresholds_accept_legacy_fields(monkeypatch) -> None:
    monkeypatch.setattr("app.promotion.service.get_settings", _settings)
    model = SimpleNamespace(metrics={"accuracy": 0.7, "macro_f1": 0.6})
    _assert_metrics_thresholds(model)


def test_metrics_thresholds_reject_low_nested_metrics(monkeypatch) -> None:
    monkeypatch.setattr("app.promotion.service.get_settings", _settings)
    model = SimpleNamespace(
        metrics={
            "classification": {
                "global": {
                    "balanced_accuracy": 0.5,
                    "macro_f1": 0.6,
                }
            }
        }
    )
    with pytest.raises(ValueError, match="accuracy below threshold"):
        _assert_metrics_thresholds(model)
