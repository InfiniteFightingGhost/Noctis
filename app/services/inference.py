from __future__ import annotations

import numpy as np

from app.ml.registry import LoadedModel


def predict_windows(
    model: LoadedModel, windows: list[np.ndarray]
) -> list[dict[str, object]]:
    if not windows:
        return []
    for window in windows:
        if window.ndim != 2:
            raise ValueError("Window tensor must be 2D")
        if window.shape[1] != model.feature_schema.size:
            raise ValueError("Window tensor has invalid feature dimension")
    batch = np.stack(windows, axis=0)
    if batch.ndim != 3:
        raise ValueError("Batch tensor must be 3D")
    probabilities = model.model.predict_proba(batch)
    predictions: list[dict[str, object]] = []
    for row in probabilities:
        max_idx = int(np.argmax(row))
        label = model.model.labels[max_idx]
        predictions.append(
            {
                "predicted_stage": label,
                "confidence": float(row[max_idx]),
                "probabilities": {
                    model.model.labels[i]: float(row[i]) for i in range(len(row))
                },
            }
        )
    return predictions
