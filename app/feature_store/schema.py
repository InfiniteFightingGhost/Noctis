from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
import uuid

from app.reproducibility.hashing import hash_json


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    dtype: str
    allowed_range: dict[str, float] | None
    description: str | None
    introduced_in_version: str
    deprecated_in_version: str | None
    position: int


@dataclass(frozen=True)
class FeatureSchemaRecord:
    id: uuid.UUID
    version: str
    hash: str
    description: str | None
    is_active: bool
    created_at: datetime
    features: list[FeatureDefinition]

    @property
    def feature_names(self) -> list[str]:
        return [feature.name for feature in self.features]

    @property
    def feature_count(self) -> int:
        return len(self.features)

    def feature_lookup(self) -> dict[str, FeatureDefinition]:
        return {feature.name: feature for feature in self.features}


def parse_feature_schema_payload(
    payload: dict[str, Any],
) -> tuple[str, str | None, list[FeatureDefinition]]:
    version = payload.get("version")
    if not version or not isinstance(version, str):
        raise ValueError("Feature schema version is required")
    description = payload.get("description")
    raw_features = payload.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        raise ValueError("Feature schema must include a non-empty features list")
    features: list[FeatureDefinition] = []
    for idx, item in enumerate(raw_features):
        if isinstance(item, str):
            name = item
            dtype = "float32"
            allowed_range = None
            feature_description = None
            introduced_in_version = version
            deprecated_in_version = None
            position = idx
        elif isinstance(item, dict):
            name_raw = item.get("name")
            if not name_raw or not isinstance(name_raw, str):
                raise ValueError("Feature name is required")
            name = name_raw
            dtype = str(item.get("dtype") or "float32")
            allowed_range_raw = item.get("allowed_range")
            if allowed_range_raw is not None and not isinstance(allowed_range_raw, dict):
                raise ValueError("allowed_range must be a JSON object")
            allowed_range = (
                cast(dict[str, float], allowed_range_raw)
                if isinstance(allowed_range_raw, dict)
                else None
            )
            feature_description = item.get("description")
            introduced_in_version = str(item.get("introduced_in_version") or version)
            deprecated_in_version = item.get("deprecated_in_version")
            if deprecated_in_version is not None:
                deprecated_in_version = str(deprecated_in_version)
            position = int(item.get("position", idx))
        else:
            raise ValueError("Feature definitions must be strings or objects")
        features.append(
            FeatureDefinition(
                name=name,
                dtype=dtype,
                allowed_range=allowed_range,
                description=feature_description,
                introduced_in_version=introduced_in_version,
                deprecated_in_version=deprecated_in_version,
                position=position,
            )
        )
    _validate_feature_definitions(features)
    ordered = sorted(features, key=lambda feature: feature.position)
    return version, description, ordered


def feature_schema_hash(features: list[FeatureDefinition]) -> str:
    payload = [
        {
            "name": feature.name,
            "dtype": feature.dtype,
            "allowed_range": feature.allowed_range,
            "description": feature.description,
            "introduced_in_version": feature.introduced_in_version,
            "deprecated_in_version": feature.deprecated_in_version,
            "position": feature.position,
        }
        for feature in features
    ]
    return hash_json(payload)


def _validate_feature_definitions(features: list[FeatureDefinition]) -> None:
    names = [feature.name for feature in features]
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique")
    positions = [feature.position for feature in features]
    if len(set(positions)) != len(positions):
        raise ValueError("Feature positions must be unique")
    sorted_positions = sorted(positions)
    expected_positions = list(range(len(features)))
    if sorted_positions != expected_positions:
        raise ValueError("Feature positions must be contiguous starting at 0")
    for feature in features:
        if not feature.dtype:
            raise ValueError("Feature dtype is required")
        if feature.allowed_range is not None:
            min_value = feature.allowed_range.get("min")
            max_value = feature.allowed_range.get("max")
            if min_value is not None and max_value is not None:
                if float(min_value) > float(max_value):
                    raise ValueError("allowed_range min must be <= max")
