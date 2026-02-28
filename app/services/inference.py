from __future__ import annotations

import numpy as np

from app.ml.decoding import viterbi_decode_probabilities_with_penalties
from app.ml.registry import LoadedModel
from app.ml.validation import prepare_batch


def _resolve_output_labels(labels: list[str], n_classes: int) -> list[str]:
    resolved = [str(label) for label in labels]
    if len(resolved) == n_classes:
        return resolved
    if len(resolved) > n_classes:
        return resolved[:n_classes]
    return resolved + [f"class_{idx}" for idx in range(len(resolved), n_classes)]


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
    expected_dim = int(expected_input_dim)
    if str(feature_strategy) == "sequence":
        expected_dim = int(model.metadata.get("base_input_dim", model.feature_schema.size))
    batch = prepare_batch(
        windows,
        feature_strategy=str(feature_strategy),
        expected_input_dim=expected_dim,
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
    output_labels = _resolve_output_labels(model.model.labels, probabilities.shape[1])
    transition_penalties = model.model.transition_penalties(output_labels)
    decoded = viterbi_decode_probabilities_with_penalties(
        probabilities,
        output_labels,
        transition_penalties=transition_penalties,
    )
    predictions: list[dict[str, object]] = []
    for idx, row in enumerate(probabilities):
        max_idx = int(decoded[idx])
        label = output_labels[max_idx]
        predictions.append(
            {
                "predicted_stage": label,
                "confidence": float(row[max_idx]),
                "probabilities": {output_labels[i]: float(row[i]) for i in range(len(row))},
            }
        )
    return predictions
