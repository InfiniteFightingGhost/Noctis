from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.ml.feature_schema import FeatureSchema, load_feature_schema
from app.ml.model import BaseModelAdapter, load_model
from app.core.metrics import MODEL_RELOAD_FAILURE, MODEL_RELOAD_SUCCESS
from app.resilience.faults import is_fault_active
from app.utils.errors import ModelUnavailableError


@dataclass(frozen=True)
class LoadedModel:
    version: str
    model: BaseModelAdapter
    feature_schema: FeatureSchema
    metadata: dict[str, Any]


class ModelRegistry:
    def __init__(self, root: Path, active_version: str) -> None:
        self._root = root
        self._active_version = active_version
        self._lock = threading.RLock()
        self._loaded: LoadedModel | None = None
        self._last_error: Exception | None = None

    def load_active(self) -> LoadedModel:
        with self._lock:
            try:
                model_dir = self._root / self._active_version
                bundle = load_model(model_dir)
                feature_schema = load_feature_schema(model_dir / "feature_schema.json")
                expected = getattr(bundle.model, "n_features_in_", None)
                if expected is not None and int(expected) != feature_schema.size:
                    raise ValueError("Model feature size mismatch")
                self._loaded = LoadedModel(
                    version=self._active_version,
                    model=bundle.model,
                    feature_schema=feature_schema,
                    metadata=bundle.metadata,
                )
                self._last_error = None
                MODEL_RELOAD_SUCCESS.inc()
                return self._loaded
            except Exception as exc:
                self._last_error = exc
                MODEL_RELOAD_FAILURE.inc()
                raise ModelUnavailableError(str(exc)) from exc

    def get_loaded(self) -> LoadedModel:
        with self._lock:
            if is_fault_active("model_unavailable"):
                raise ModelUnavailableError("Model unavailable (fault injected)")
            if self._loaded is None:
                return self.load_active()
            return self._loaded

    def reload(self) -> LoadedModel:
        with self._lock:
            return self.load_active()

    @property
    def last_error(self) -> Exception | None:
        return self._last_error
