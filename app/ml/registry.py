from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.ml.feature_schema import FeatureSchema, load_feature_schema
from app.ml.model import LinearSoftmaxModel, load_artifacts


@dataclass(frozen=True)
class LoadedModel:
    version: str
    model: LinearSoftmaxModel
    feature_schema: FeatureSchema
    metadata: dict[str, Any]


class ModelRegistry:
    def __init__(self, root: Path, active_version: str) -> None:
        self._root = root
        self._active_version = active_version
        self._lock = threading.RLock()
        self._loaded: LoadedModel | None = None

    def load_active(self) -> LoadedModel:
        with self._lock:
            model_dir = self._root / self._active_version
            artifacts = load_artifacts(model_dir)
            feature_schema = load_feature_schema(model_dir / "feature_schema.json")
            if artifacts.weights.shape[0] != feature_schema.size:
                raise ValueError("Model feature size mismatch")
            metadata = json.loads((model_dir / "metadata.json").read_text())
            self._loaded = LoadedModel(
                version=self._active_version,
                model=LinearSoftmaxModel(artifacts),
                feature_schema=feature_schema,
                metadata=metadata,
            )
            return self._loaded

    def get_loaded(self) -> LoadedModel:
        with self._lock:
            if self._loaded is None:
                return self.load_active()
            return self._loaded

    def reload(self) -> LoadedModel:
        return self.load_active()
