from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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
class CnnBiLstmModelConfig:
    use_dataset_conditioning: bool = False
    conditioning_mode: str = "onehot"
    conditioning_embed_dim: int = 8
    conv_channels: list[int] = field(default_factory=lambda: [64, 128])
    conv_kernel_size: int = 3
    conv_dropout: float = 0.1
    lstm_hidden_size: int = 128
    lstm_layers: int = 1
    lstm_dropout: float = 0.1
    head_hidden_dims: list[int] = field(default_factory=lambda: [128])
    head_dropout: float = 0.2


@dataclass(frozen=True)
class CnnBiLstmTrainingConfig:
    batch_size: int = 64
    max_epochs: int = 30
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.0
    loss_type: str = "weighted_ce"
    focal_gamma: float = 2.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    scheduler_type: str = "plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-6
    gradient_clip_norm: float = 1.0
    instability_macro_f1_threshold: float = 0.15
    n2_class_weight_multiplier: float = 1.0
    enable_domain_transfer_tests: bool = False


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
    model: CnnBiLstmModelConfig = CnnBiLstmModelConfig()
    training: CnnBiLstmTrainingConfig = CnnBiLstmTrainingConfig()


def load_training_config(path: Path) -> TrainingConfig:
    payload = _load_payload(path)
    return training_config_from_payload(payload)


def training_config_from_payload(payload: dict[str, Any]) -> TrainingConfig:
    search_payload = payload.get("search", {})
    model_payload = payload.get("model", {})
    training_payload = payload.get("training", {})
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
        model=CnnBiLstmModelConfig(
            use_dataset_conditioning=bool(model_payload.get("use_dataset_conditioning", False)),
            conditioning_mode=str(model_payload.get("conditioning_mode", "onehot")),
            conditioning_embed_dim=int(model_payload.get("conditioning_embed_dim", 8)),
            conv_channels=[int(value) for value in model_payload.get("conv_channels", [64, 128])],
            conv_kernel_size=int(model_payload.get("conv_kernel_size", 3)),
            conv_dropout=float(model_payload.get("conv_dropout", 0.1)),
            lstm_hidden_size=int(model_payload.get("lstm_hidden_size", 128)),
            lstm_layers=int(model_payload.get("lstm_layers", 1)),
            lstm_dropout=float(model_payload.get("lstm_dropout", 0.1)),
            head_hidden_dims=[int(value) for value in model_payload.get("head_hidden_dims", [128])],
            head_dropout=float(model_payload.get("head_dropout", 0.2)),
        ),
        training=CnnBiLstmTrainingConfig(
            batch_size=int(training_payload.get("batch_size", 64)),
            max_epochs=int(training_payload.get("max_epochs", 30)),
            early_stopping_patience=int(training_payload.get("early_stopping_patience", 6)),
            early_stopping_min_delta=float(training_payload.get("early_stopping_min_delta", 0.0)),
            loss_type=str(training_payload.get("loss_type", "weighted_ce")),
            focal_gamma=float(training_payload.get("focal_gamma", 2.0)),
            learning_rate=float(training_payload.get("learning_rate", 1e-3)),
            weight_decay=float(training_payload.get("weight_decay", 1e-4)),
            optimizer_beta1=float(training_payload.get("optimizer_beta1", 0.9)),
            optimizer_beta2=float(training_payload.get("optimizer_beta2", 0.999)),
            scheduler_type=str(training_payload.get("scheduler_type", "plateau")),
            scheduler_factor=float(training_payload.get("scheduler_factor", 0.5)),
            scheduler_patience=int(training_payload.get("scheduler_patience", 3)),
            scheduler_min_lr=float(training_payload.get("scheduler_min_lr", 1e-6)),
            gradient_clip_norm=float(training_payload.get("gradient_clip_norm", 1.0)),
            instability_macro_f1_threshold=float(
                training_payload.get("instability_macro_f1_threshold", 0.15)
            ),
            n2_class_weight_multiplier=float(
                training_payload.get("n2_class_weight_multiplier", 1.0)
            ),
            enable_domain_transfer_tests=bool(
                training_payload.get("enable_domain_transfer_tests", False)
            ),
        ),
    )
    validate_training_config(config)
    return config


def validate_training_config(config: TrainingConfig) -> None:
    if config.model_type not in {"gradient_boosting", "cnn_bilstm"}:
        raise ValueError("Unsupported model_type")
    if config.model_type == "cnn_bilstm":
        if config.feature_strategy != "sequence":
            raise ValueError("cnn_bilstm requires feature_strategy='sequence'")
        if config.model.conditioning_mode not in {"onehot", "embedding"}:
            raise ValueError("conditioning_mode must be onehot or embedding")
        if config.model.conditioning_embed_dim <= 0:
            raise ValueError("conditioning_embed_dim must be > 0")
        if not config.model.conv_channels or any(v <= 0 for v in config.model.conv_channels):
            raise ValueError("conv_channels must contain positive integers")
        if config.model.conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be > 0")
        if config.model.lstm_hidden_size <= 0:
            raise ValueError("lstm_hidden_size must be > 0")
        if config.model.lstm_layers <= 0:
            raise ValueError("lstm_layers must be > 0")
        if any(v <= 0 for v in config.model.head_hidden_dims):
            raise ValueError("head_hidden_dims must contain positive integers")
        if config.training.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if config.training.max_epochs <= 0:
            raise ValueError("max_epochs must be > 0")
        if config.training.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0")
        if config.training.loss_type not in {"weighted_ce", "focal"}:
            raise ValueError("loss_type must be weighted_ce or focal")
        if config.training.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if config.training.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if config.training.scheduler_type not in {"none", "plateau"}:
            raise ValueError("scheduler_type must be none or plateau")
        if config.training.scheduler_type == "plateau":
            if config.training.scheduler_factor <= 0 or config.training.scheduler_factor >= 1:
                raise ValueError("scheduler_factor must be in (0, 1)")
            if config.training.scheduler_patience < 0:
                raise ValueError("scheduler_patience must be >= 0")
        if config.training.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be > 0")
        if config.training.instability_macro_f1_threshold < 0:
            raise ValueError("instability_macro_f1_threshold must be >= 0")
        if config.training.n2_class_weight_multiplier <= 0:
            raise ValueError("n2_class_weight_multiplier must be > 0")


def cnn_model_payload(config: TrainingConfig) -> dict[str, Any]:
    return asdict(config.model)


def cnn_training_payload(config: TrainingConfig) -> dict[str, Any]:
    return asdict(config.training)


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
