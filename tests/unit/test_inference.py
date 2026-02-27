from __future__ import annotations

import numpy as np

from app.services.inference import predict_windows


class _FakeModel:
    def __init__(self) -> None:
        self.labels = ["WAKE", "N1", "N2", "N3", "REM"]
        self.last_dataset_ids = None

    def predict_proba(self, batch: np.ndarray, *, dataset_ids=None) -> np.ndarray:
        self.last_dataset_ids = dataset_ids
        return np.tile(
            np.asarray([[0.9, 0.05, 0.03, 0.01, 0.01]], dtype=np.float32), (batch.shape[0], 1)
        )


class _FakeLoaded:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.metadata = {
            "feature_strategy": "sequence",
            "expected_input_dim": 3,
            "window_size": 2,
            "inference_dataset_id": "CAP",
        }

        class _Schema:
            size = 3

        self.feature_schema = _Schema()


def test_predict_windows_passes_dataset_ids_for_sequence() -> None:
    loaded = _FakeLoaded()
    windows = [np.zeros((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32)]
    predictions = predict_windows(loaded, windows)
    assert len(predictions) == 2
    assert loaded.model.last_dataset_ids.tolist() == ["CAP", "CAP"]
