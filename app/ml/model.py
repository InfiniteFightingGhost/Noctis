from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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
    label_map = cast(list[str], json.loads((model_dir / "label_map.json").read_text()))
    return ModelArtifacts(weights=weights, bias=bias, label_map=label_map)


@dataclass(frozen=True)
class ModelBundle:
    model: "BaseModelAdapter"
    metadata: dict[str, Any]


class BaseModelAdapter:
    @property
    def labels(self) -> list[str]:
        raise NotImplementedError

    def predict_proba(
        self,
        batch: np.ndarray,
        *,
        dataset_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class LinearSoftmaxModel(BaseModelAdapter):
    def __init__(self, artifacts: ModelArtifacts) -> None:
        self._weights = artifacts.weights
        self._bias = artifacts.bias
        self._label_map = artifacts.label_map

    @property
    def labels(self) -> list[str]:
        return self._label_map

    def predict_proba(
        self,
        batch: np.ndarray,
        *,
        dataset_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        if batch.ndim != 2:
            raise ValueError("Model expects 2D feature batch")
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

    def predict_proba(
        self,
        batch: np.ndarray,
        *,
        dataset_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        if batch.ndim != 2:
            raise ValueError("Model expects 2D feature batch")
        if self._scaler is not None:
            batch = self._scaler.transform(batch)
        return self._model.predict_proba(batch)

    @property
    def n_features_in_(self) -> int | None:
        return getattr(self._model, "n_features_in_", None)


class TorchSequenceModelAdapter(BaseModelAdapter):
    def __init__(
        self,
        *,
        model_dir: Path,
        metadata: dict[str, Any],
        label_map: list[str],
    ) -> None:
        torch = importlib.import_module("torch")
        training_payload = cast(
            dict[str, Any],
            json.loads((model_dir / "training_config.json").read_text()),
        )
        model_cfg = cast(dict[str, Any], training_payload.get("model", {}))
        if not model_cfg:
            raise ValueError("CNN model config missing from training_config.json")
        self._torch = torch
        self._label_map = label_map
        self._dataset_id_map = cast(
            dict[str, int],
            metadata.get(
                "dataset_id_map",
                {"DODH": 0, "CAP": 1, "ISRUC": 2, "UNKNOWN": 3},
            ),
        )

        from app.training.cnn_bilstm import CnnBiLstmNetwork

        input_dim = int(metadata.get("expected_input_dim") or 0)
        if input_dim <= 0:
            raise ValueError("CNN metadata expected_input_dim missing")
        self._network = CnnBiLstmNetwork(
            input_dim=input_dim,
            num_classes=len(label_map),
            num_domains=max(len(self._dataset_id_map), 1),
            model_cfg=model_cfg,
            torch=torch,
            nn=torch.nn,
        )
        state = torch.load(model_dir / "model.pt", map_location="cpu")
        self._network.module.load_state_dict(state)
        self._network.module.eval()

        scaler_path = model_dir / "scaler.json"
        if scaler_path.exists():
            scaler_payload = cast(dict[str, Any], json.loads(scaler_path.read_text()))
            self._scaler_mean = np.asarray(scaler_payload.get("mean", []), dtype=np.float32)
            self._scaler_std = np.asarray(scaler_payload.get("std", []), dtype=np.float32)
        else:
            self._scaler_mean = None
            self._scaler_std = None

    @property
    def labels(self) -> list[str]:
        return self._label_map

    def predict_proba(
        self,
        batch: np.ndarray,
        *,
        dataset_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        if batch.ndim != 3:
            raise ValueError("Sequence model expects 3D feature batch")
        x = np.asarray(batch, dtype=np.float32)
        if (
            self._scaler_mean is not None
            and self._scaler_std is not None
            and self._scaler_mean.size > 0
        ):
            x = (x - self._scaler_mean.reshape(1, 1, -1)) / self._scaler_std.reshape(1, 1, -1)
        if dataset_ids is None:
            dataset_ids = np.full(x.shape[0], "UNKNOWN", dtype=object)
        dataset_idx = np.asarray(
            [
                self._dataset_id_map.get(str(value), self._dataset_id_map.get("UNKNOWN", 0))
                for value in dataset_ids
            ],
            dtype=np.int64,
        )

        torch = self._torch
        with torch.no_grad():
            logits = self._network(
                torch.from_numpy(x).float(),
                torch.from_numpy(dataset_idx).long(),
            )
            probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()


def load_model(model_dir: Path) -> ModelBundle:
    metadata_path = model_dir / "metadata.json"
    metadata = (
        cast(dict[str, Any], json.loads(metadata_path.read_text()))
        if metadata_path.exists()
        else {}
    )
    adapter: BaseModelAdapter
    if (model_dir / "model.bin").exists():
        model = joblib.load(model_dir / "model.bin")
        scaler = None
        if (model_dir / "scaler.bin").exists():
            scaler = joblib.load(model_dir / "scaler.bin")
        label_map_path = model_dir / "label_map.json"
        if label_map_path.exists():
            label_map = cast(list[str], json.loads(label_map_path.read_text()))
        else:
            label_map = cast(list[str], metadata.get("label_map", []))
        model_labels = [str(label) for label in getattr(model, "classes_", [])]
        if model_labels and label_map and model_labels != label_map:
            raise ValueError("Model label_map ordering mismatch")
        adapter = SklearnModelAdapter(model=model, scaler=scaler, label_map=label_map)
        return ModelBundle(model=adapter, metadata=metadata)
    if (model_dir / "model.pt").exists():
        label_map_path = model_dir / "label_map.json"
        if label_map_path.exists():
            label_map = cast(list[str], json.loads(label_map_path.read_text()))
        else:
            label_map = cast(list[str], metadata.get("label_map", []))
        adapter = TorchSequenceModelAdapter(
            model_dir=model_dir,
            metadata=metadata,
            label_map=label_map,
        )
        return ModelBundle(model=adapter, metadata=metadata)
    artifacts = load_artifacts(model_dir)
    adapter = LinearSoftmaxModel(artifacts)
    return ModelBundle(model=adapter, metadata=metadata)
