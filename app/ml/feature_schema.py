from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    features: list[str]

    @property
    def size(self) -> int:
        return len(self.features)


def load_feature_schema(path: Path) -> FeatureSchema:
    payload = json.loads(path.read_text())
    version = payload.get("version")
    features = payload.get("features")
    if not version or not isinstance(features, list):
        raise ValueError("Invalid feature schema")
    return FeatureSchema(version=version, features=features)
