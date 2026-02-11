from __future__ import annotations

import base64
from typing import Any

import numpy as np

from app.ml.feature_schema import FeatureSchema


def decode_features(payload: Any, schema: FeatureSchema) -> np.ndarray:
    if isinstance(payload, list):
        vector = np.asarray(payload, dtype=np.float32)
    elif isinstance(payload, dict):
        missing = [name for name in schema.features if name not in payload]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        vector = np.asarray(
            [payload[name] for name in schema.features], dtype=np.float32
        )
    elif isinstance(payload, str):
        raw = base64.b64decode(payload)
        vector = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError("Unsupported feature payload")

    if vector.shape[0] != schema.size:
        raise ValueError(f"Expected {schema.size} features, got {vector.shape[0]}")
    return vector
