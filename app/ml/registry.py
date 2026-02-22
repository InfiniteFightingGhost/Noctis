from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.ml.feature_schema import FeatureSchema, load_feature_schema
from app.ml.model import BaseModelAdapter, load_model
from app.core.metrics import MODEL_RELOAD_FAILURE, MODEL_RELOAD_SUCCESS
from app.resilience.faults import is_fault_active
from app.reproducibility.hashing import hash_artifact_dir
from app.utils.errors import ModelUnavailableError


@dataclass(frozen=True)
class LoadedModel:
    version: str
    model: BaseModelAdapter
    feature_schema: FeatureSchema
    metadata: dict[str, Any]


class ModelRegistry:
    def __init__(
        self,
        root: Path,
        active_version: str,
        *,
        schema_provider: Callable[[], Any] | None = None,
    ) -> None:
        self._root = root
        self._active_version = active_version
        self._schema_provider = schema_provider
        self._lock = threading.RLock()
        self._loaded: LoadedModel | None = None
        self._last_error: Exception | None = None

    def load_active(self) -> LoadedModel:
        with self._lock:
            try:
                model_dir = self._root / self._active_version
                bundle = load_model(model_dir)
                feature_schema = load_feature_schema(model_dir / "feature_schema.json")
                expected_hash = bundle.metadata.get("artifact_hash")
                if expected_hash:
                    computed_hash = hash_artifact_dir(
                        model_dir,
                        exclude_files={"metadata.json"},
                    )
                    if computed_hash != expected_hash:
                        raise ValueError("Model artifact hash mismatch")
                feature_strategy = bundle.metadata.get("feature_strategy")
                expected_input_dim = bundle.metadata.get("expected_input_dim")
                if feature_strategy is None or expected_input_dim is None:
                    raise ValueError("Model feature strategy metadata missing")
                expected = getattr(bundle.model, "n_features_in_", None)
                if expected is not None and int(expected) != int(expected_input_dim):
                    raise ValueError("Model feature size mismatch")
                if self._schema_provider is not None:
                    active_schema = self._schema_provider()
                    if active_schema is None:
                        raise ValueError("Active feature schema missing")
                    if active_schema.version != feature_schema.version:
                        raise ValueError("Model feature schema mismatch")
                    if (
                        feature_schema.schema_hash
                        and active_schema.hash
                        and feature_schema.schema_hash != active_schema.hash
                    ):
                        raise ValueError("Model feature schema hash mismatch")
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
