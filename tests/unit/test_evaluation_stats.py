from __future__ import annotations

from app.evaluation.stats import (
    accuracy,
    average_confidence,
    build_labels,
    confidence_histogram,
    confusion_matrix,
    entropy_metrics,
    merge_labels,
    night_summary_metrics,
    per_class_metrics,
    per_class_frequency,
    prediction_distribution,
    transition_matrix,
)


def test_confusion_metrics() -> None:
    truth = ["W", "N1", "N1", "N2"]
    pred = ["W", "N2", "N1", "N2"]
    labels = ["N1", "N2", "W"]
    matrix = confusion_matrix(truth, pred, labels)
    metrics = per_class_metrics(matrix, labels)
    assert accuracy(matrix) == 0.75
    assert metrics[0]["label"] == "N1"


def test_confidence_histogram_entropy() -> None:
    histogram = confidence_histogram([0.1, 0.2, 0.95], bins=5)
    assert histogram["counts"] == [1, 1, 0, 0, 1]
    entropy = entropy_metrics(
        [
            {"W": 0.5, "N1": 0.5},
            {"W": 1.0, "N1": 0.0},
        ]
    )
    assert entropy["mean"] > 0.0


def test_transition_matrix() -> None:
    stages = ["W", "N1", "N2", "N2", "W"]
    labels = ["N1", "N2", "W"]
    matrix = transition_matrix(stages, labels)
    assert matrix[labels.index("W")][labels.index("N1")] == 1
    assert matrix[labels.index("N2")][labels.index("W")] == 1


def test_build_labels_sorted_unique() -> None:
    labels = build_labels(["N2", "W", "W"], ["N1", "W"])
    assert labels == ["N1", "N2", "W"]


def test_confidence_histogram_edge_values() -> None:
    histogram = confidence_histogram([0.0, 1.0], bins=4)
    assert histogram["counts"] == [1, 0, 0, 1]


def test_entropy_metrics_single_row() -> None:
    entropy = entropy_metrics([{"W": 0.25, "N1": 0.75}])
    assert entropy["p50"] == entropy["mean"]
    assert entropy["p90"] == entropy["mean"]


def test_transition_matrix_ignores_unknowns() -> None:
    stages = ["W", "X", "N1"]
    labels = ["N1", "W"]
    matrix = transition_matrix(stages, labels)
    assert matrix == [[0, 0], [0, 0]]


def test_distribution_and_summary() -> None:
    stages = ["W", "N1", "N2", "W"]
    distribution = prediction_distribution(stages)
    frequency = per_class_frequency(stages)
    summary = night_summary_metrics(stages)
    assert distribution["W"] == 0.5
    assert frequency[0]["count"] > 0
    assert summary["total_minutes"] > 0.0


def test_average_confidence() -> None:
    assert average_confidence([0.5, 0.75]) == 0.625


def test_entropy_metrics_skips_non_finite() -> None:
    entropy = entropy_metrics([{"W": 0.5, "N1": float("nan")}])
    assert entropy["mean"] >= 0.0


def test_confidence_histogram_skips_non_finite() -> None:
    histogram = confidence_histogram([0.2, float("inf")], bins=4)
    assert histogram["counts"] == [1, 0, 0, 0]


def test_merge_labels_preserves_order() -> None:
    merged = merge_labels(["W", "N1"], ["N2", "W", "REM"])
    assert merged == ["W", "N1", "N2", "REM"]
