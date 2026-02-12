from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from app.ml.feature_schema import FeatureSchema, load_feature_schema
from app.training.config import TrainingConfig


@dataclass(frozen=True)
class TrainingResult:
    version: str
    artifact_dir: Path
    metrics: dict[str, Any]
    label_map: list[str]


def train_model(
    *,
    config: TrainingConfig,
    version: str,
) -> TrainingResult:
    dataset = _load_dataset(config.dataset_dir)
    feature_schema = load_feature_schema(config.feature_schema_path)
    X, y, label_map = _prepare_features(dataset, config, feature_schema)
    splits = _extract_splits(dataset)
    train_idx = splits.get("train")
    val_idx = splits.get("val")
    test_idx = splits.get("test")
    if train_idx is None or len(train_idx) == 0:
        raise ValueError("Training split is empty")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    y_train = y[train_idx]
    sample_weight = _compute_sample_weights(y_train, config.class_balance)
    estimator = GradientBoostingClassifier(random_state=config.random_seed)
    estimator = _search_hyperparameters(
        estimator,
        X_train,
        y_train,
        sample_weight,
        config,
    )
    estimator.fit(X_train, y_train, sample_weight=sample_weight)
    eval_idx = test_idx if test_idx is not None and len(test_idx) > 0 else val_idx
    if eval_idx is None or len(eval_idx) == 0:
        eval_idx = train_idx
    X_eval = scaler.transform(X[eval_idx])
    y_eval = y[eval_idx]
    y_pred = estimator.predict(X_eval)
    metrics = _evaluate_metrics(y_eval, y_pred, label_map)
    metrics["evaluation_split"] = "test" if eval_idx is test_idx else "val"
    artifact_dir = config.output_root / version
    artifact_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(estimator, artifact_dir / "model.bin")
    joblib.dump(scaler, artifact_dir / "scaler.bin")
    (artifact_dir / "feature_schema.json").write_text(
        json.dumps(
            {"version": feature_schema.version, "features": feature_schema.features},
            indent=2,
        )
    )
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (artifact_dir / "training_config.json").write_text(
        json.dumps(_config_payload(config), indent=2)
    )
    (artifact_dir / "label_map.json").write_text(json.dumps(label_map, indent=2))
    metadata = _build_metadata(config, version, dataset, label_map, feature_schema)
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return TrainingResult(
        version=version,
        artifact_dir=artifact_dir,
        metrics=metrics,
        label_map=label_map,
    )


def _load_dataset(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "dataset.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return dict(np.load(path, allow_pickle=True))


def _prepare_features(
    dataset: dict[str, Any],
    config: TrainingConfig,
    feature_schema: FeatureSchema,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = dataset["X"]
    y = dataset["y"]
    if config.feature_strategy == "mean":
        if X.ndim != 3:
            raise ValueError("Expected 3D window tensor for mean strategy")
        X = X.mean(axis=1)
        if X.shape[1] != feature_schema.size:
            raise ValueError("Feature schema size mismatch")
    elif config.feature_strategy == "flatten":
        X = X.reshape(X.shape[0], -1)
    else:
        raise ValueError("Unknown feature strategy")
    label_map = [str(label) for label in dataset.get("label_map", np.unique(y))]
    return X.astype(np.float32), y.astype(str), label_map


def _extract_splits(dataset: dict[str, Any]) -> dict[str, np.ndarray | None]:
    return {
        "train": dataset.get("split_train"),
        "val": dataset.get("split_val"),
        "test": dataset.get("split_test"),
    }


def _compute_sample_weights(y: np.ndarray, balance: str) -> np.ndarray | None:
    if balance == "none":
        return None
    unique, counts = np.unique(y, return_counts=True)
    weights = {label: 1.0 / count for label, count in zip(unique, counts)}
    return np.asarray([weights[label] for label in y], dtype=np.float32)


def _search_hyperparameters(
    estimator: GradientBoostingClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    config: TrainingConfig,
) -> GradientBoostingClassifier:
    if not config.search.param_grid:
        return estimator.set_params(**config.hyperparameters)
    cv = StratifiedKFold(
        n_splits=config.search.cv_folds, shuffle=True, random_state=config.random_seed
    )
    fit_params = {}
    if sample_weight is not None:
        fit_params = {"sample_weight": sample_weight}
    if config.search.method == "grid":
        search = GridSearchCV(
            estimator,
            param_grid=config.search.param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
        )
    else:
        search = RandomizedSearchCV(
            estimator,
            param_distributions=config.search.param_grid,
            n_iter=config.search.n_iter,
            scoring="f1_macro",
            cv=cv,
            random_state=config.random_seed,
            n_jobs=-1,
        )
    search.fit(X_train, y_train, **fit_params)
    return search.best_estimator_


def _evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: list[str],
) -> dict[str, Any]:
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_map,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_map)
    transitions_true = _transition_matrix(y_true, label_map)
    transitions_pred = _transition_matrix(y_pred, label_map)
    stability = np.abs(transitions_true - transitions_pred).sum()
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_precision": {
            label: float(value) for label, value in zip(label_map, precision)
        },
        "per_class_recall": {
            label: float(value) for label, value in zip(label_map, recall)
        },
        "confusion_matrix": matrix.tolist(),
        "transition_matrix": transitions_pred.tolist(),
        "transition_matrix_stability": float(stability),
    }


def _transition_matrix(values: np.ndarray, label_map: list[str]) -> np.ndarray:
    index = {label: idx for idx, label in enumerate(label_map)}
    matrix = np.zeros((len(label_map), len(label_map)), dtype=np.float32)
    if len(values) < 2:
        return matrix
    for prev, curr in zip(values[:-1], values[1:]):
        if prev not in index or curr not in index:
            continue
        matrix[index[prev], index[curr]] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums


def _build_metadata(
    config: TrainingConfig,
    version: str,
    dataset: dict[str, Any],
    label_map: list[str],
    feature_schema: FeatureSchema,
) -> dict[str, Any]:
    return {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.random_seed,
        "dataset_snapshot": _load_dataset_metadata(config.dataset_dir),
        "feature_schema_version": feature_schema.version,
        "model_type": config.model_type,
        "label_map": label_map,
        "git_commit_hash": _git_commit_hash(),
    }


def _load_dataset_metadata(dataset_dir: Path) -> dict[str, Any] | None:
    path = dataset_dir / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _config_payload(config: TrainingConfig) -> dict[str, Any]:
    return {
        "dataset_dir": str(config.dataset_dir),
        "output_root": str(config.output_root),
        "feature_schema_path": str(config.feature_schema_path),
        "model_type": config.model_type,
        "random_seed": config.random_seed,
        "class_balance": config.class_balance,
        "feature_strategy": config.feature_strategy,
        "hyperparameters": config.hyperparameters,
        "search": {
            "method": config.search.method,
            "param_grid": config.search.param_grid,
            "n_iter": config.search.n_iter,
            "cv_folds": config.search.cv_folds,
        },
        "version_bump": config.version_bump,
        "experiment_name": config.experiment_name,
        "metrics_thresholds": config.metrics_thresholds,
    }
