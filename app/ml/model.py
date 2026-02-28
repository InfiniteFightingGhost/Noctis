from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np

from app.training.mmwave import engineer_mmwave_features


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

    def transition_penalties(self, labels: list[str]) -> np.ndarray | None:
        return None


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
                {"DODH": 0, "CAP": 1, "ISRUC": 2, "SLEEP-EDF": 3, "UNKNOWN": 4},
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
        state_payload = torch.load(model_dir / "model.pt", map_location="cpu")
        state_dict = _extract_state_dict(state_payload)
        checkpoint_format = _detect_sequence_checkpoint_format(state_dict)
        compatible_state = _build_compatible_sequence_state_dict(
            state_dict,
            checkpoint_format=checkpoint_format,
        )
        strict = checkpoint_format == "new"
        self._network.module.load_state_dict(compatible_state, strict=strict)
        self._network.module.eval()

        scaler_path = model_dir / "scaler.json"
        if scaler_path.exists():
            scaler_payload = cast(dict[str, Any], json.loads(scaler_path.read_text()))
            self._scaler_mean = np.asarray(scaler_payload.get("mean", []), dtype=np.float32)
            self._scaler_std = np.asarray(scaler_payload.get("std", []), dtype=np.float32)
            self._scaler_policy = str(scaler_payload.get("policy", "global_zscore"))
        else:
            self._scaler_mean = None
            self._scaler_std = None
            self._scaler_policy = "global_zscore"
        temperature_path = model_dir / "temperature.json"
        if temperature_path.exists():
            temp_payload = cast(dict[str, Any], json.loads(temperature_path.read_text()))
            self._temperature = float(temp_payload.get("temperature", 1.0))
        else:
            self._temperature = 1.0
        feature_pipeline_path = model_dir / "feature_pipeline.json"
        if feature_pipeline_path.exists():
            self._feature_pipeline = cast(
                dict[str, Any], json.loads(feature_pipeline_path.read_text())
            )
        else:
            self._feature_pipeline = cast(dict[str, Any], metadata.get("feature_pipeline", {}))
        transition_matrix_path = model_dir / "transition_matrix.json"
        self._transition_penalties: np.ndarray | None = None
        self._transition_labels = [str(label) for label in label_map]
        if transition_matrix_path.exists():
            payload = cast(dict[str, Any], json.loads(transition_matrix_path.read_text()))
            penalties = payload.get("penalties")
            if isinstance(penalties, list):
                parsed = np.asarray(penalties, dtype=np.float32)
                if parsed.shape == (len(label_map), len(label_map)):
                    self._transition_penalties = parsed

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
        x = preprocess_sequence_batch(
            batch=np.asarray(batch, dtype=np.float32),
            feature_pipeline=self._feature_pipeline,
            scaler_mean=self._scaler_mean,
            scaler_std=self._scaler_std,
            scaler_policy=self._scaler_policy,
        )
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
            logits = logits / max(self._temperature, 1e-6)
            probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def transition_penalties(self, labels: list[str]) -> np.ndarray | None:
        if self._transition_penalties is None:
            return None
        requested = [str(label) for label in labels]
        if requested == self._transition_labels:
            return self._transition_penalties
        index = {label: idx for idx, label in enumerate(self._transition_labels)}
        if any(label not in index for label in requested):
            return None
        order = [index[label] for label in requested]
        return self._transition_penalties[np.ix_(order, order)]


def preprocess_sequence_batch(
    *,
    batch: np.ndarray,
    feature_pipeline: dict[str, Any],
    scaler_mean: np.ndarray | None,
    scaler_std: np.ndarray | None,
    scaler_policy: str,
) -> np.ndarray:
    x = np.asarray(batch, dtype=np.float32)
    base_features = feature_pipeline.get("base_feature_schema")
    if isinstance(base_features, list) and base_features:
        x, _, _ = engineer_mmwave_features(
            x,
            feature_names=[str(v) for v in base_features],
            low_agreement_threshold=float(feature_pipeline.get("low_agreement_threshold", 0.5)),
            eps=float(feature_pipeline.get("eps", 1e-6)),
        )
    if scaler_mean is not None and scaler_std is not None and scaler_mean.size > 0:
        if scaler_policy == "recording_zscore_then_global":
            local_mean = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=(0, 1))
            local_std = np.nanstd(np.where(np.isfinite(x), x, np.nan), axis=(0, 1))
            local_mean = np.where(np.isfinite(local_mean), local_mean, 0.0)
            local_std = np.where(np.isfinite(local_std) & (local_std > 1e-8), local_std, 1.0)
            x = (x - local_mean.reshape(1, 1, -1)) / local_std.reshape(1, 1, -1)
        mean = scaler_mean.reshape(1, 1, -1)
        std = scaler_std.reshape(1, 1, -1)
        finite = np.isfinite(x)
        if not finite.all():
            replacement = np.broadcast_to(mean, x.shape)
            x = x.copy()
            x[~finite] = replacement[~finite]
        x = (x - mean) / std
    return x.astype(np.float32)


def _extract_state_dict(state_payload: Any) -> dict[str, Any]:
    if not isinstance(state_payload, dict):
        raise ValueError("Unsupported model.pt payload")
    nested = state_payload.get("state_dict")
    if isinstance(nested, dict):
        raw_state = nested
    else:
        raw_state = state_payload
    return {str(key): value for key, value in raw_state.items()}


def _detect_sequence_checkpoint_format(state_dict: dict[str, Any]) -> str:
    has_primary = any(
        key.startswith("primary_head.") or key.startswith("module.primary_head.")
        for key in state_dict
    )
    has_aux = any(
        key.startswith("aux_head.") or key.startswith("module.aux_head.") for key in state_dict
    )
    has_legacy_head = any(
        key.startswith("head.") or key.startswith("module.head.") for key in state_dict
    )
    if has_primary or has_aux:
        return "new"
    if has_legacy_head:
        return "legacy"
    raise ValueError("Unsupported sequence checkpoint format")


def _build_compatible_sequence_state_dict(
    state_dict: dict[str, Any],
    *,
    checkpoint_format: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized = key[7:] if key.startswith("module.") else key
        if checkpoint_format == "legacy" and normalized.startswith("head."):
            normalized = f"primary_head.{normalized[len('head.'):]}"
        out[normalized] = value
    return out


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
