from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


class LinearSoftmaxModel:
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
