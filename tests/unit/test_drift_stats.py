from __future__ import annotations

from app.drift.router import _classify_feature, _classify_metric
from app.drift.stats import kl_divergence, mean, psi, std, z_score


def test_psi_kl_zero_when_equal() -> None:
    dist = {"W": 0.5, "N1": 0.5}
    assert psi(dist, dist) == 0.0
    assert kl_divergence(dist, dist) == 0.0


def test_z_score() -> None:
    assert z_score(10.0, 10.0, 2.0) == 0.0


def test_z_score_zero_std() -> None:
    assert z_score(10.0, 8.0, 0.0) == 0.0


def test_mean_std_empty() -> None:
    assert mean([]) == 0.0
    assert std([]) == 0.0


def test_mean_std_values() -> None:
    values = [1.0, 3.0, 5.0]
    assert mean(values) == 3.0
    assert std(values) == 2.0


def test_psi_kl_normalizes_zero_distribution() -> None:
    current = {"W": 0.0, "N1": 0.0}
    baseline = {"W": 0.0, "N1": 0.0}
    assert psi(current, baseline) == 0.0
    assert kl_divergence(current, baseline) == 0.0


def test_classify_metric_severity() -> None:
    status, severity, score = _classify_metric(
        {"psi": 0.5, "kl_divergence": None, "z_score": None},
        {"psi": 0.2, "kl": 0.1, "z": 3.0},
    )
    assert status == "alert"
    assert severity in {"MEDIUM", "HIGH"}
    assert score == 0.5


def test_classify_feature_severity() -> None:
    status, severity, score = _classify_feature(
        {"z_score": 6.1},
        {"psi": 0.2, "kl": 0.1, "z": 3.0},
    )
    assert status == "alert"
    assert severity == "HIGH"
    assert score == 6.1
