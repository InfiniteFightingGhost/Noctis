from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

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

from app.db.session import run_with_db_retry
from app.feature_store.service import get_feature_schema_by_version
from app.lineage.service import build_lineage_metadata
from app.ml.feature_schema import FeatureSchema, load_feature_schema
from app.reproducibility.hashing import hash_artifact_dir, hash_json
from app.reproducibility.snapshots import verify_snapshot_checksum
from app.training.config import TrainingConfig, cnn_model_payload, cnn_training_payload
from app.training.mmwave import FOUR_CLASS_LABELS, remap_stage_label, validate_base_feature_schema


@dataclass(frozen=True)
class TrainingResult:
    version: str
    artifact_dir: Path
    metrics: dict[str, Any]
    label_map: list[str]
    dataset_snapshot_id: str
    feature_schema_version: str
    git_commit_hash: str | None
    training_seed: int
    metrics_hash: str
    artifact_hash: str


def train_model(
    *,
    config: TrainingConfig,
    version: str,
) -> TrainingResult:
    dataset = _load_dataset(config.dataset_dir)
    dataset_metadata = _load_dataset_metadata(config.dataset_dir)
    _validate_split_policy(dataset_metadata, config)
    label_strategy = _label_strategy(dataset_metadata)
    _validate_label_strategy(label_strategy, config)
    snapshot_id = _resolve_snapshot_id(config, dataset_metadata)
    feature_schema_path = _resolve_feature_schema_path(config)
    feature_schema = load_feature_schema(feature_schema_path)
    schema_record = _resolve_feature_schema_record(feature_schema, dataset_metadata)
    _verify_snapshot_integrity(snapshot_id, schema_record, feature_schema, dataset_metadata)
    splits = _extract_splits(dataset)
    train_idx = splits.get("train")
    val_idx = splits.get("val")
    test_idx = splits.get("test")
    if train_idx is None or len(train_idx) == 0:
        raise ValueError("Training split is empty")
    artifact_dir = config.output_root / version
    artifact_dir.mkdir(parents=True, exist_ok=False)
    _assert_split_recording_isolation(dataset, splits)
    split_manifest = _build_split_manifest(dataset, splits)
    (artifact_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2))

    model_classes: list[str]
    metrics: dict[str, Any]
    estimator: Any | None = None
    expected_input_dim_override: int | None = None
    if config.model_type == "cnn_bilstm":
        from app.training.cnn_bilstm import train_cnn_bilstm

        feature_names = [str(name) for name in feature_schema.features]
        X_seq, y, label_map = _prepare_sequence_features(
            dataset,
            config,
            feature_names=feature_names,
        )
        dataset_ids = _dataset_ids(dataset, len(y))
        recording_ids = _recording_ids(dataset, len(y))
        eval_split = _resolve_evaluation_split_name(config, test_idx, val_idx, label_strategy)
        train_eval_split = eval_split
        if train_eval_split is None:
            train_eval_split = "val" if val_idx is not None and len(val_idx) > 0 else "test"
        output = train_cnn_bilstm(
            config=config,
            artifact_dir=artifact_dir,
            X=X_seq,
            y=y,
            label_map=label_map,
            feature_names=feature_names,
            dataset_ids=dataset_ids,
            recording_ids=recording_ids,
            splits=splits,
            evaluation_split_name=train_eval_split,
        )
        metrics = output.metrics
        if eval_split is None:
            metrics = {
                "evaluation_split": "none",
                "evaluation_disabled": True,
            }
        model_classes = output.model_classes
        expected_input_dim_override = int(output.metrics.get("input_dim", X_seq.shape[2]))
    elif config.model_type == "gradient_boosting":
        X, y, label_map = _prepare_features(dataset, config, feature_schema)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        y_train = y[train_idx]
        _ensure_finite("X_train", X_train)
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
        model_label_map = [str(label) for label in estimator.classes_]
        if set(model_label_map) != set(label_map):
            raise ValueError("Estimator classes do not match dataset labels")
        if config.enforce_label_map_order and model_label_map != label_map:
            raise ValueError("Estimator label order does not match dataset label_map")
        label_map = model_label_map
        eval_split = _resolve_evaluation_split_name(config, test_idx, val_idx, label_strategy)
        if eval_split is None:
            metrics = {
                "evaluation_split": "none",
                "evaluation_disabled": True,
            }
        else:
            eval_idx = test_idx if eval_split == "test" else val_idx
            if eval_idx is None or len(eval_idx) == 0:
                raise ValueError("Evaluation split is empty")
            X_eval = scaler.transform(X[eval_idx])
            _ensure_finite("X_eval", X_eval)
            y_eval = y[eval_idx]
            y_pred = estimator.predict(X_eval)
            metrics = _evaluate_metrics(y_eval, y_pred, label_map)
            metrics["evaluation_split"] = eval_split
        joblib.dump(estimator, artifact_dir / "model.bin")
        joblib.dump(scaler, artifact_dir / "scaler.bin")
        model_classes = [str(label) for label in estimator.classes_]
    else:
        raise ValueError("Unsupported model_type")

    metrics_hash = hash_json(metrics)
    schema_payload = {
        "id": str(schema_record.id),
        "version": schema_record.version,
        "hash": schema_record.hash,
        "features": [
            {
                "name": feature.name,
                "dtype": feature.dtype,
                "allowed_range": feature.allowed_range,
                "description": feature.description,
                "introduced_in_version": feature.introduced_in_version,
                "deprecated_in_version": feature.deprecated_in_version,
                "position": feature.position,
            }
            for feature in schema_record.features
        ],
    }
    (artifact_dir / "feature_schema.json").write_text(json.dumps(schema_payload, indent=2))
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (artifact_dir / "training_config.json").write_text(
        json.dumps(_config_payload(config), indent=2)
    )
    label_map_order_hash = hash_json(model_classes)
    (artifact_dir / "label_map.json").write_text(json.dumps(model_classes, indent=2))
    artifact_hash = hash_artifact_dir(artifact_dir, exclude_files={"metadata.json"})
    git_commit_hash = _git_commit_hash()
    metadata = _build_metadata(
        config,
        version,
        dataset_metadata,
        model_classes,
        feature_schema,
        model_classes,
        snapshot_id=snapshot_id,
        feature_schema_hash=schema_record.hash,
        metrics_hash=metrics_hash,
        artifact_hash=artifact_hash,
        git_commit_hash=git_commit_hash,
        label_map_order_hash=label_map_order_hash,
        expected_input_dim_override=expected_input_dim_override,
        base_input_dim=feature_schema.size,
    )
    if config.model_type == "cnn_bilstm":
        metadata["feature_pipeline"] = metrics.get("feature_pipeline", {})
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return TrainingResult(
        version=version,
        artifact_dir=artifact_dir,
        metrics=metrics,
        label_map=model_classes,
        dataset_snapshot_id=snapshot_id,
        feature_schema_version=feature_schema.version,
        git_commit_hash=git_commit_hash,
        training_seed=config.random_seed,
        metrics_hash=metrics_hash,
        artifact_hash=artifact_hash,
    )


def _load_dataset(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "dataset.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return dict(np.load(path, allow_pickle=True))


def _resolve_feature_schema_path(config: TrainingConfig) -> Path:
    if config.feature_schema_path:
        return config.feature_schema_path
    fallback = config.dataset_dir / "feature_schema.json"
    if not fallback.exists():
        raise FileNotFoundError(fallback)
    return fallback


def _resolve_snapshot_id(config: TrainingConfig, dataset_metadata: dict[str, Any] | None) -> str:
    snapshot_id = config.dataset_snapshot_id
    if not snapshot_id and dataset_metadata:
        snapshot_id = dataset_metadata.get("dataset_snapshot_id")
    if not snapshot_id:
        raise ValueError("Training requires a dataset snapshot reference")
    try:
        return str(uuid.UUID(str(snapshot_id)))
    except ValueError as exc:
        raise ValueError("Invalid dataset snapshot id") from exc


def _resolve_feature_schema_record(
    feature_schema: FeatureSchema,
    dataset_metadata: dict[str, Any] | None,
):
    record = run_with_db_retry(
        lambda session: get_feature_schema_by_version(session, feature_schema.version),
        operation_name="training_feature_schema",
    )
    if record is None:
        raise ValueError("Feature schema not registered")
    if record.feature_names != feature_schema.features:
        raise ValueError("Feature ordering mismatch")
    if dataset_metadata:
        expected_version = dataset_metadata.get("feature_schema_version")
        if expected_version and expected_version != record.version:
            raise ValueError("Feature schema version mismatch")
    return record


def _verify_snapshot_integrity(
    snapshot_id: str,
    schema_record,
    feature_schema: FeatureSchema,
    dataset_metadata: dict[str, Any] | None,
) -> None:
    snapshot_uuid = uuid.UUID(snapshot_id)

    def _verify(session):
        result = verify_snapshot_checksum(session, snapshot_id=snapshot_uuid)
        if not result.matches:
            raise ValueError("Dataset snapshot checksum mismatch")
        return result

    run_with_db_retry(
        _verify,
        operation_name="verify_snapshot_checksum",
    )
    if feature_schema.schema_hash and feature_schema.schema_hash != schema_record.hash:
        raise ValueError("Feature schema hash mismatch")
    if dataset_metadata:
        stored_hash = dataset_metadata.get("feature_schema_hash")
        if stored_hash and stored_hash != schema_record.hash:
            raise ValueError("Dataset snapshot schema hash mismatch")


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
    label_map = _validate_label_map(label_map, y, enforce_order=config.enforce_label_map_order)
    X = X.astype(np.float32)
    y = y.astype(str)
    _ensure_finite("X", X)
    return X, y, label_map


def _prepare_sequence_features(
    dataset: dict[str, Any],
    config: TrainingConfig,
    *,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = dataset["X"]
    y = dataset["y"]
    if X.ndim != 3:
        raise ValueError("Expected 3D window tensor for sequence strategy")
    if config.feature_strategy != "sequence":
        raise ValueError("cnn_bilstm requires sequence feature strategy")
    validate_base_feature_schema(feature_names)
    remapped: list[str] = []
    for label in y:
        mapped = remap_stage_label(label)
        if mapped is None:
            raise ValueError(f"Unsupported stage label for unified 4-class pipeline: {label}")
        remapped.append(mapped)
    X = X.astype(np.float32)
    y = np.asarray(remapped, dtype=str)
    return X, y, list(FOUR_CLASS_LABELS)


def _dataset_ids(dataset: dict[str, Any], row_count: int) -> np.ndarray:
    values = dataset.get("dataset_ids")
    if values is None:
        return np.full(row_count, "UNKNOWN", dtype=object)
    dataset_ids = np.asarray(values).astype(str)
    if dataset_ids.shape[0] != row_count:
        raise ValueError("dataset_ids length does not match X")
    return dataset_ids


def _recording_ids(dataset: dict[str, Any], row_count: int) -> np.ndarray:
    values = dataset.get("recording_ids")
    if values is None:
        return np.asarray([f"recording_{idx}" for idx in range(row_count)], dtype=object)
    recording_ids = np.asarray(values).astype(str)
    if recording_ids.shape[0] != row_count:
        raise ValueError("recording_ids length does not match X")
    return recording_ids


def _extract_splits(dataset: dict[str, Any]) -> dict[str, np.ndarray | None]:
    return {
        "train": dataset.get("split_train"),
        "val": dataset.get("split_val"),
        "test": dataset.get("split_test"),
    }


def _resolve_evaluation_split_name(
    config: TrainingConfig,
    test_idx: np.ndarray | None,
    val_idx: np.ndarray | None,
    label_strategy: str | None,
) -> str | None:
    if label_strategy and label_strategy != "ground_truth_only":
        return None
    if config.evaluation_split_policy == "none":
        return None
    if config.evaluation_split_policy == "test_only":
        if test_idx is None or len(test_idx) == 0:
            raise ValueError("Evaluation split is empty")
        return "test"
    if config.evaluation_split_policy == "val_or_test":
        if test_idx is not None and len(test_idx) > 0:
            return "test"
        if val_idx is not None and len(val_idx) > 0:
            return "val"
        raise ValueError("Evaluation split is empty")
    raise ValueError("Unknown evaluation split policy")


def _assert_split_recording_isolation(
    dataset: dict[str, Any],
    splits: dict[str, np.ndarray | None],
) -> None:
    recording_ids = dataset.get("recording_ids")
    if recording_ids is None:
        return
    recordings = np.asarray(recording_ids).astype(str)
    seen: dict[str, str] = {}
    for split_name, indices in splits.items():
        if indices is None:
            continue
        for idx in indices:
            recording = recordings[int(idx)]
            existing = seen.get(recording)
            if existing and existing != split_name:
                raise ValueError("Split leakage detected for recording_id")
            seen[recording] = split_name


def _build_split_manifest(
    dataset: dict[str, Any],
    splits: dict[str, np.ndarray | None],
) -> dict[str, Any]:
    raw_recording_ids = dataset.get("recording_ids")
    has_recording_ids = raw_recording_ids is not None
    recording_ids = (
        np.asarray(raw_recording_ids).astype(str)
        if has_recording_ids
        else np.asarray([], dtype=str)
    )
    payload: dict[str, Any] = {
        "split_sizes": {},
        "recording_counts": {},
        "recording_overlap": False,
        "recording_ids_present": has_recording_ids,
    }
    split_recordings: dict[str, set[str]] = {}
    for split_name, indices in splits.items():
        if indices is None:
            payload["split_sizes"][split_name] = 0
            payload["recording_counts"][split_name] = 0
            split_recordings[split_name] = set()
            continue
        if has_recording_ids:
            values = {recording_ids[int(idx)] for idx in indices}
        else:
            values = {f"row_{int(idx)}" for idx in indices}
        split_recordings[split_name] = values
        payload["split_sizes"][split_name] = int(len(indices))
        payload["recording_counts"][split_name] = int(len(values))
    names = list(split_recordings.keys())
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if split_recordings[left].intersection(split_recordings[right]):
                payload["recording_overlap"] = True
    return payload


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
        "per_class_precision": {label: float(value) for label, value in zip(label_map, precision)},
        "per_class_recall": {label: float(value) for label, value in zip(label_map, recall)},
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
    dataset_metadata: dict[str, Any] | None,
    label_map: list[str],
    feature_schema: FeatureSchema,
    model_classes: list[str],
    *,
    snapshot_id: str,
    feature_schema_hash: str,
    metrics_hash: str,
    artifact_hash: str,
    git_commit_hash: str | None,
    label_map_order_hash: str,
    expected_input_dim_override: int | None,
    base_input_dim: int,
) -> dict[str, Any]:
    dataset_metadata = dataset_metadata or {}
    window_size = int(dataset_metadata.get("window_size") or 0)
    epoch_seconds = int(dataset_metadata.get("epoch_seconds") or 0)
    if window_size <= 0:
        raise ValueError("Dataset window_size missing from metadata")
    expected_input_dim = (
        expected_input_dim_override
        if expected_input_dim_override is not None
        else _expected_input_dim(feature_schema.size, config.feature_strategy, window_size)
    )
    return {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.random_seed,
        "dataset_snapshot": dataset_metadata,
        "dataset_snapshot_id": snapshot_id,
        "feature_schema_version": feature_schema.version,
        "feature_schema_hash": feature_schema_hash,
        "model_type": config.model_type,
        "label_map": label_map,
        "label_map_source": "estimator_classes",
        "label_map_order_hash": label_map_order_hash,
        "model_classes": model_classes,
        "git_commit_hash": git_commit_hash,
        "metrics_hash": metrics_hash,
        "artifact_hash": artifact_hash,
        "feature_strategy": config.feature_strategy,
        "window_size": window_size,
        "epoch_seconds": epoch_seconds,
        "window_stride": int(dataset_metadata.get("window_stride") or 1),
        "expected_input_dim": expected_input_dim,
        "base_input_dim": int(base_input_dim),
        "dataset_id_map": {"DODH": 0, "CAP": 1, "ISRUC": 2, "SLEEP-EDF": 3, "UNKNOWN": 4},
        "inference_dataset_id": "UNKNOWN",
        "split_grouped_stratification": config.split_grouped_stratification,
        "split_stratify_key": config.split_stratify_key,
        "lineage": build_lineage_metadata(
            dataset_snapshot_id=snapshot_id,
            feature_schema_version=feature_schema.version,
            feature_schema_hash=feature_schema_hash,
            training_seed=config.random_seed,
            git_commit_hash=git_commit_hash,
            metrics_hash=metrics_hash,
            artifact_hash=artifact_hash,
        ),
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
        "feature_schema_path": str(config.feature_schema_path)
        if config.feature_schema_path
        else None,
        "dataset_snapshot_id": config.dataset_snapshot_id,
        "model_type": config.model_type,
        "random_seed": config.random_seed,
        "class_balance": config.class_balance,
        "feature_strategy": config.feature_strategy,
        "allow_predicted_labels": config.allow_predicted_labels,
        "split_strategy": config.split_strategy,
        "split_seed": config.split_seed,
        "split_grouping_key": config.split_grouping_key,
        "split_time_aware": config.split_time_aware,
        "split_grouped_stratification": config.split_grouped_stratification,
        "split_stratify_key": config.split_stratify_key,
        "evaluation_split_policy": config.evaluation_split_policy,
        "enforce_label_map_order": config.enforce_label_map_order,
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
        "model": cnn_model_payload(config),
        "training": cnn_training_payload(config),
        "evaluation": {
            "calibration_bins": config.evaluation.calibration_bins,
            "epoch_seconds": config.evaluation.epoch_seconds,
            "forbidden_transitions": [
                [int(src), int(dst)] for src, dst in config.evaluation.forbidden_transitions
            ],
            "hard_thresholds": config.evaluation.hard_thresholds,
            "soft_thresholds": config.evaluation.soft_thresholds,
        },
    }


def _validate_split_policy(dataset_metadata: dict[str, Any] | None, config: TrainingConfig) -> None:
    if not dataset_metadata:
        raise ValueError("Dataset metadata is required for training")
    policy = dataset_metadata.get("split_policy") or {}
    if not policy:
        raise ValueError("Dataset split policy missing")
    expected = {
        "split_strategy": config.split_strategy,
        "seed": config.split_seed,
        "grouping_key": config.split_grouping_key,
        "time_aware": config.split_time_aware,
        "grouped_stratification": config.split_grouped_stratification,
        "stratify_key": config.split_stratify_key,
    }
    defaults = {
        "grouped_stratification": False,
        "stratify_key": "dataset_rem_bucket",
    }
    for key, value in expected.items():
        observed = policy.get(key, defaults.get(key))
        if observed != value:
            raise ValueError("Dataset split policy mismatch")


def _label_strategy(dataset_metadata: dict[str, Any] | None) -> str | None:
    if not dataset_metadata:
        return None
    return dataset_metadata.get("label_source_policy") or dataset_metadata.get("label_strategy")


def _validate_label_strategy(label_strategy: str | None, config: TrainingConfig) -> None:
    if label_strategy == "predicted_only":
        raise ValueError("Predicted-only labels are not allowed for training")
    if label_strategy and label_strategy != "ground_truth_only":
        if not config.allow_predicted_labels:
            raise ValueError("Predicted labels are disabled for training")


def _validate_label_map(
    label_map: list[str],
    y: np.ndarray,
    *,
    enforce_order: bool,
) -> list[str]:
    if not label_map:
        raise ValueError("Dataset label_map is empty")
    unique_labels = sorted({str(label) for label in y})
    if sorted(set(label_map)) != unique_labels:
        raise ValueError("Dataset label_map labels do not match y")
    if enforce_order and label_map != unique_labels:
        raise ValueError("Dataset label_map ordering mismatch")
    return list(label_map)


def _expected_input_dim(feature_size: int, feature_strategy: str, window_size: int) -> int:
    if feature_strategy == "mean":
        return feature_size
    if feature_strategy == "flatten":
        return feature_size * max(window_size, 1)
    if feature_strategy == "sequence":
        return feature_size
    raise ValueError("Unknown feature strategy")


def _ensure_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
