from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import uuid


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    features: list[str]
    schema_id: uuid.UUID | None = None
    schema_hash: str | None = None

    @property
    def size(self) -> int:
        return len(self.features)


def load_feature_schema(path: Path) -> FeatureSchema:
    payload = json.loads(path.read_text())
    version = payload.get("version")
    features_payload = payload.get("features")
    if not version or not isinstance(features_payload, list):
        raise ValueError("Invalid feature schema")
    features: list[str] = []
    for item in features_payload:
        if isinstance(item, str):
            features.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            features.append(item["name"])
        else:
            raise ValueError("Invalid feature schema features")
    schema_id = payload.get("id") or payload.get("schema_id")
    schema_hash = payload.get("hash") or payload.get("schema_hash")
    parsed_id = uuid.UUID(schema_id) if schema_id else None
    return FeatureSchema(
        version=version,
        features=features,
        schema_id=parsed_id,
        schema_hash=schema_hash,
    )
