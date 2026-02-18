from __future__ import annotations

import base64
from typing import Any

import numpy as np

from app.feature_store.schema import FeatureSchemaRecord, FeatureDefinition


def validate_feature_payload(
    payload: Any, schema: FeatureSchemaRecord
) -> list[float] | str:
    normalized = _unwrap_features(payload)
    if isinstance(normalized, list):
        return _validate_list(normalized, schema)
    if isinstance(normalized, dict):
        return _validate_mapping(normalized, schema)
    if isinstance(normalized, str):
        return _validate_base64(normalized, schema)
    raise ValueError("features must be list, dict, or base64 string")


def _unwrap_features(payload: Any) -> Any:
    if isinstance(payload, dict) and "features" in payload and len(payload) == 1:
        return payload.get("features")
    return payload


def _validate_list(values: list[Any], schema: FeatureSchemaRecord) -> list[float]:
    if len(values) != schema.feature_count:
        raise ValueError(f"Expected {schema.feature_count} features, got {len(values)}")
    validated: list[float] = []
    for value, definition in zip(values, schema.features, strict=True):
        validated.append(_validate_value(value, definition))
    return validated


def _validate_mapping(
    values: dict[str, Any], schema: FeatureSchemaRecord
) -> list[float]:
    expected = set(schema.feature_names)
    provided = set(values.keys())
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    if missing:
        raise ValueError(f"Missing features: {missing}")
    if extra:
        raise ValueError(f"Unknown features: {extra}")
    validated: list[float] = []
    for definition in schema.features:
        validated.append(_validate_value(values[definition.name], definition))
    return validated


def _validate_base64(payload: str, schema: FeatureSchemaRecord) -> str:
    try:
        raw = base64.b64decode(payload)
    except Exception as exc:
        raise ValueError("Invalid base64 feature payload") from exc
    vector = np.frombuffer(raw, dtype=np.float32)
    if vector.shape[0] != schema.feature_count:
        raise ValueError(
            f"Expected {schema.feature_count} features, got {vector.shape[0]}"
        )
    for value, definition in zip(vector.tolist(), schema.features, strict=True):
        _validate_value(value, definition)
    return payload


def _validate_value(value: Any, definition: FeatureDefinition) -> float:
    if isinstance(value, bool):
        raise ValueError("Feature values must be numeric")
    dtype = definition.dtype.lower()
    if dtype.startswith("int"):
        if not isinstance(value, int):
            raise ValueError(f"Feature {definition.name} must be int")
        numeric = float(value)
    else:
        if not isinstance(value, (int, float)):
            raise ValueError(f"Feature {definition.name} must be numeric")
        numeric = float(value)
    allowed = definition.allowed_range or {}
    min_value = allowed.get("min")
    max_value = allowed.get("max")
    if min_value is not None and numeric < float(min_value):
        raise ValueError(f"Feature {definition.name} below min range")
    if max_value is not None and numeric > float(max_value):
        raise ValueError(f"Feature {definition.name} above max range")
    return numeric
