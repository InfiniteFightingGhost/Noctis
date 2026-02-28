from __future__ import annotations

import numpy as np

from app.eval.night_evaluator import per_night_eval


def test_per_night_eval_outputs_temporal_and_clinical_metrics() -> None:
    y_true = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    y_pred = np.asarray([0, 1, 2, 2, 0, 1, 2, 3], dtype=np.int64)
    proba = np.asarray(
        [
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.05, 0.05, 0.7, 0.2],
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.75, 0.1, 0.05],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ],
        dtype=np.float64,
    )
    recording_id = np.asarray(["n2", "n2", "n2", "n2", "n1", "n1", "n1", "n1"], dtype=object)
    dataset_id = np.asarray(["DS", "DS", "DS", "DS", "DS", "DS", "DS", "DS"], dtype=object)

    report = per_night_eval(
        y_true,
        y_pred,
        proba,
        recording_id,
        dataset_id,
        class_names=["W", "Light", "Deep", "REM"],
        calibration_bins=5,
    )

    assert [row["recording_id"] for row in report["nights"]] == ["n1", "n2"]
    assert "transition_matrix_error_norm" in report["nights"][0]
    assert "wake_deep_direct_rate" in report["nights"][0]
    assert "hypnogram_edit_distance_raw" in report["nights"][0]
    assert "clinical_error" in report["nights"][0]
