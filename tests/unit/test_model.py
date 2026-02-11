from __future__ import annotations

from pathlib import Path

import numpy as np

from app.ml.model import LinearSoftmaxModel, load_artifacts


def test_model_predict_proba() -> None:
    artifacts = load_artifacts(Path("models/active"))
    model = LinearSoftmaxModel(artifacts)
    batch = np.zeros((2, 10), dtype=np.float32)
    proba = model.predict_proba(batch)
    assert proba.shape == (2, 5)
    assert np.allclose(proba.sum(axis=1), 1.0)
