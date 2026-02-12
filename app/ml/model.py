from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass(frozen=True)
class ModelArtifacts:
    weights: np.ndarray
    bias: np.ndarray
    label_map: list[str]


def load_artifacts(model_dir: Path) -> ModelArtifacts:
    weights = np.load(model_dir / "weights.npy")
    bias = np.load(model_dir / "bias.npy")
    label_map = json.loads((model_dir / "label_map.json").read_text())
    return ModelArtifacts(weights=weights, bias=bias, label_map=label_map)


@dataclass(frozen=True)
class ModelBundle:
    model: "BaseModelAdapter"
    metadata: dict[str, Any]


class BaseModelAdapter:
    @property
    def labels(self) -> list[str]:
        raise NotImplementedError

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearSoftmaxModel(BaseModelAdapter):
    def __init__(self, artifacts: ModelArtifacts) -> None:
        self._weights = artifacts.weights
        self._bias = artifacts.bias
        self._label_map = artifacts.label_map

    @property
    def labels(self) -> list[str]:
        return self._label_map

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        if batch.ndim == 3:
            batch = batch.mean(axis=1)
        logits = batch @ self._weights + self._bias
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


class SklearnModelAdapter(BaseModelAdapter):
    def __init__(self, model: Any, scaler: Any | None, label_map: list[str]) -> None:
        self._model = model
        self._scaler = scaler
        self._label_map = label_map

    @property
    def labels(self) -> list[str]:
        return self._label_map

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        if batch.ndim == 3:
            batch = batch.mean(axis=1)
        if self._scaler is not None:
            batch = self._scaler.transform(batch)
        return self._model.predict_proba(batch)

    @property
    def n_features_in_(self) -> int | None:
        return getattr(self._model, "n_features_in_", None)


def load_model(model_dir: Path) -> ModelBundle:
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if (model_dir / "model.bin").exists():
        model = joblib.load(model_dir / "model.bin")
        scaler = None
        if (model_dir / "scaler.bin").exists():
            scaler = joblib.load(model_dir / "scaler.bin")
        label_map_path = model_dir / "label_map.json"
        if label_map_path.exists():
            label_map = json.loads(label_map_path.read_text())
        else:
            label_map = metadata.get("label_map", [])
        adapter = SklearnModelAdapter(model=model, scaler=scaler, label_map=label_map)
        return ModelBundle(model=adapter, metadata=metadata)
    artifacts = load_artifacts(model_dir)
    adapter = LinearSoftmaxModel(artifacts)
    return ModelBundle(model=adapter, metadata=metadata)
