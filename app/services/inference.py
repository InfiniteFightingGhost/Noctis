from __future__ import annotations

import numpy as np

from app.ml.registry import LoadedModel
from app.ml.validation import prepare_batch


def predict_windows(
    model: LoadedModel,
    windows: list[np.ndarray],
    *,
    dataset_id: str | None = None,
) -> list[dict[str, object]]:
    if not windows:
        return []
    feature_strategy = model.metadata.get("feature_strategy")
    expected_input_dim = model.metadata.get("expected_input_dim")
    window_size = model.metadata.get("window_size")
    if feature_strategy is None or expected_input_dim is None or window_size is None:
        raise ValueError("Model metadata missing feature strategy")
    batch = prepare_batch(
        windows,
        feature_strategy=str(feature_strategy),
        expected_input_dim=int(expected_input_dim),
        feature_dim=model.feature_schema.size,
        window_size=int(window_size),
    )
    if str(feature_strategy) == "sequence":
        effective_dataset_id = dataset_id or str(
            model.metadata.get("inference_dataset_id", "UNKNOWN")
        )
        dataset_ids = np.full(batch.shape[0], effective_dataset_id, dtype=object)
        probabilities = model.model.predict_proba(batch, dataset_ids=dataset_ids)
    else:
        probabilities = model.model.predict_proba(batch)
    if not np.isfinite(probabilities).all():
        raise ValueError("Model probabilities contain NaN or Inf")
    predictions: list[dict[str, object]] = []
    for row in probabilities:
        max_idx = int(np.argmax(row))
        label = model.model.labels[max_idx]
        predictions.append(
            {
                "predicted_stage": label,
                "confidence": float(row[max_idx]),
                "probabilities": {model.model.labels[i]: float(row[i]) for i in range(len(row))},
            }
        )
    return predictions
