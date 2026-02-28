from __future__ import annotations

import numpy as np

from app.eval.calibration import calibration_metrics


def test_calibration_reports_named_per_class_and_reliability_bins() -> None:
    y_true = np.asarray([0, 1, 0, 1], dtype=np.int64)
    proba = np.asarray(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.55, 0.45],
            [0.7, 0.3],
        ],
        dtype=np.float64,
    )

    metrics = calibration_metrics(y_true, proba, n_bins=4, class_names=["WAKE", "N1"])

    assert metrics["brier"] >= 0.0
    assert len(metrics["reliability"]["bins"]) == 4
    assert "WAKE" in metrics["per_class"]
    assert "gap" in metrics["reliability"]["bins"][0]
