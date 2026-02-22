from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingSearchConfig:
    method: str = "random"
    param_grid: dict[str, Any] = field(default_factory=dict)
    n_iter: int = 10
    cv_folds: int = 5


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: Path
    output_root: Path
    feature_schema_path: Path | None
    dataset_snapshot_id: str | None = None
    model_type: str = "gradient_boosting"
    random_seed: int = 42
    class_balance: str = "none"
    feature_strategy: str = "mean"
    allow_predicted_labels: bool = False
    split_strategy: str = "recording"
    split_seed: int = 42
    split_grouping_key: str = "recording_id"
    split_time_aware: bool = False
    evaluation_split_policy: str = "val_or_test"
    enforce_label_map_order: bool = True
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    search: TrainingSearchConfig = TrainingSearchConfig()
    version_bump: str = "patch"
    experiment_name: str | None = None
    metrics_thresholds: dict[str, float] = field(default_factory=dict)


def load_training_config(path: Path) -> TrainingConfig:
    payload = _load_payload(path)
    return training_config_from_payload(payload)


def training_config_from_payload(payload: dict[str, Any]) -> TrainingConfig:
    search_payload = payload.get("search", {})
    split_strategy = payload.get("split_strategy", "recording")
    config = TrainingConfig(
        dataset_dir=Path(payload["dataset_dir"]),
        output_root=Path(payload.get("output_root", "models")),
        feature_schema_path=Path(payload["feature_schema_path"])
        if payload.get("feature_schema_path")
        else None,
        dataset_snapshot_id=payload.get("dataset_snapshot_id"),
        model_type=payload.get("model_type", "gradient_boosting"),
        random_seed=int(payload.get("random_seed", 42)),
        class_balance=payload.get("class_balance", "none"),
        feature_strategy=payload.get("feature_strategy", "mean"),
        allow_predicted_labels=bool(payload.get("allow_predicted_labels", False)),
        split_strategy=split_strategy,
        split_seed=int(payload.get("split_seed", payload.get("random_seed", 42))),
        split_grouping_key=payload.get("split_grouping_key", "recording_id"),
        split_time_aware=bool(payload.get("split_time_aware", split_strategy == "recording_time")),
        evaluation_split_policy=payload.get("evaluation_split_policy", "val_or_test"),
        enforce_label_map_order=bool(payload.get("enforce_label_map_order", True)),
        hyperparameters=dict(payload.get("hyperparameters", {})),
        search=TrainingSearchConfig(
            method=search_payload.get("method", "random"),
            param_grid=dict(search_payload.get("param_grid", {})),
            n_iter=int(search_payload.get("n_iter", 10)),
            cv_folds=int(search_payload.get("cv_folds", 5)),
        ),
        version_bump=payload.get("version_bump", "patch"),
        experiment_name=payload.get("experiment_name"),
        metrics_thresholds=dict(payload.get("metrics_thresholds", {})),
    )
    return config


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text()
    if path.suffix.lower() in {".yml", ".yaml"}:
        payload = yaml.safe_load(raw)
    else:
        payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a JSON/YAML object")
    return payload
