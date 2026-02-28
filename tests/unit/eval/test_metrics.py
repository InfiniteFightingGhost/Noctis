from __future__ import annotations

import numpy as np

from app.eval.metrics import evaluate_classification


def test_evaluate_classification_includes_specificity_and_core_scores() -> None:
    y_true = np.asarray([0, 1, 1, 2, 2, 2], dtype=np.int64)
    y_pred = np.asarray([0, 1, 0, 2, 1, 2], dtype=np.int64)

    metrics = evaluate_classification(y_true, y_pred, num_classes=3, class_names=["A", "B", "C"])

    assert metrics["macro_f1"] > 0.0
    assert metrics["weighted_f1"] > 0.0
    assert "specificity" in metrics["per_class"]["A"]
    assert metrics["support_total"] == 6
