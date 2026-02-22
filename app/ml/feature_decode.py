from __future__ import annotations

import base64
from typing import Any

import numpy as np

from app.ml.feature_schema import FeatureSchema


def decode_features(payload: Any, schema: FeatureSchema | Any) -> np.ndarray:
    payload = _unwrap_features(payload)
    feature_names = _schema_feature_names(schema)
    if isinstance(payload, list):
        vector = np.asarray(payload, dtype=np.float32)
    elif isinstance(payload, dict):
        missing = [name for name in feature_names if name not in payload]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        vector = np.asarray([payload[name] for name in feature_names], dtype=np.float32)
    elif isinstance(payload, str):
        raw = base64.b64decode(payload)
        vector = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError("Unsupported feature payload")

    if vector.shape[0] != len(feature_names):
        raise ValueError(f"Expected {len(feature_names)} features, got {vector.shape[0]}")
    if not np.isfinite(vector).all():
        raise ValueError("Decoded features contain NaN or Inf")
    return vector


def _unwrap_features(payload: Any) -> Any:
    if isinstance(payload, dict) and "features" in payload and len(payload) == 1:
        return payload.get("features")
    return payload


def _schema_feature_names(schema: Any) -> list[str]:
    if hasattr(schema, "feature_names"):
        return list(schema.feature_names)
    features = getattr(schema, "features", [])
    if features and hasattr(features[0], "name"):
        return [feature.name for feature in features]
    return list(features)
