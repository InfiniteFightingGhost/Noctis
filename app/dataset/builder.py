from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.dataset.config import DatasetBuildConfig
from app.dataset.io import export_hdf5, export_parquet, save_npz
from app.db.models import Epoch, Prediction, Recording
from app.db.session import run_with_db_retry
from app.ml.feature_decode import decode_features
from app.ml.feature_schema import FeatureSchema, load_feature_schema
from app.services.windowing import WindowedEpoch, build_windows


@dataclass(frozen=True)
class DatasetBuildResult:
    X: np.ndarray
    y: np.ndarray
    label_map: list[str]
    window_end_ts: np.ndarray
    recording_ids: np.ndarray
    splits: dict[str, np.ndarray]
    metadata: dict[str, Any]


def build_dataset(config: DatasetBuildConfig) -> DatasetBuildResult:
    feature_schema = load_feature_schema(config.feature_schema_path)

    def _op(session: Session) -> DatasetBuildResult:
        epochs = _fetch_epochs(session, config, feature_schema)
        windows, window_meta = _build_windows(epochs, config)
        labels = _fetch_labels(session, config, window_meta)
        X, y, window_end_ts, recording_ids = _align_labels(
            windows,
            window_meta,
            labels,
            label_strategy=config.label_strategy,
        )
        if X.size == 0:
            raise ValueError("No labeled windows found for dataset")
        X, y, window_end_ts, recording_ids = _balance_classes(
            X,
            y,
            window_end_ts,
            recording_ids,
            strategy=config.balance_strategy,
            seed=config.random_seed,
        )
        splits = _stratified_split_indices(
            y,
            train_ratio=config.split.train,
            val_ratio=config.split.val,
            test_ratio=config.split.test,
            seed=config.random_seed,
        )
        label_map = sorted({label for label in y})
        metadata = _build_metadata(config, feature_schema, label_map, splits)
        result = DatasetBuildResult(
            X=X,
            y=y,
            label_map=label_map,
            window_end_ts=window_end_ts,
            recording_ids=recording_ids,
            splits=splits,
            metadata=metadata,
        )
        _export_dataset(result, config, feature_schema)
        return result

    return run_with_db_retry(_op, operation_name="dataset_build")


def _fetch_epochs(
    session: Session,
    config: DatasetBuildConfig,
    feature_schema: FeatureSchema,
) -> dict[object, list[WindowedEpoch]]:
    query = session.query(
        Epoch.recording_id,
        Epoch.epoch_index,
        Epoch.epoch_start_ts,
        Epoch.feature_schema_version,
        Epoch.features_payload,
    ).join(Recording, Epoch.recording_id == Recording.id)
    filters = config.filters
    if filters.device_id:
        query = query.filter(Recording.device_id == filters.device_id)
    if filters.tenant_id:
        query = query.filter(Recording.tenant_id == filters.tenant_id)
    if filters.recording_id:
        query = query.filter(Epoch.recording_id == filters.recording_id)
    if filters.from_ts:
        query = query.filter(Epoch.epoch_start_ts >= filters.from_ts)
    if filters.to_ts:
        query = query.filter(Epoch.epoch_start_ts <= filters.to_ts)
    if filters.feature_schema_version:
        query = query.filter(
            Epoch.feature_schema_version == filters.feature_schema_version
        )
    rows = query.order_by(
        Epoch.recording_id, Epoch.epoch_index, Epoch.epoch_start_ts
    ).all()
    epochs: dict[object, list[WindowedEpoch]] = {}
    for recording_id, epoch_index, epoch_start_ts, schema_version, payload in rows:
        if schema_version != feature_schema.version:
            raise ValueError(
                f"Feature schema mismatch: expected {feature_schema.version}, got {schema_version}"
            )
        vector = decode_features(payload, feature_schema)
        key = recording_id
        epochs.setdefault(key, []).append(
            WindowedEpoch(
                epoch_index=epoch_index,
                epoch_start_ts=epoch_start_ts,
                features=vector,
            )
        )
    return epochs


def _build_windows(
    epochs: dict[object, list[WindowedEpoch]],
    config: DatasetBuildConfig,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    if not epochs:
        return [], []
    tensors: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for recording_id, items in sorted(epochs.items(), key=lambda item: str(item[0])):
        windows = build_windows(
            items,
            window_size=config.window_size,
            allow_padding=config.allow_padding,
        )
        for window in windows:
            tensors.append(window.tensor)
            meta.append(
                {
                    "recording_id": recording_id,
                    "window_end_ts": window.end_ts,
                }
            )
    return tensors, meta


def _fetch_labels(
    session: Session,
    config: DatasetBuildConfig,
    window_meta: list[dict[str, Any]],
) -> dict[tuple[str, datetime], dict[str, Any]]:
    filters = config.filters
    recording_ids = {
        meta.get("recording_id") for meta in window_meta if meta.get("recording_id")
    }
    query = session.query(
        Prediction.recording_id,
        Prediction.window_end_ts,
        Prediction.predicted_stage,
        Prediction.ground_truth_stage,
    ).join(Recording, Prediction.recording_id == Recording.id)
    if filters.recording_id:
        query = query.filter(Prediction.recording_id == filters.recording_id)
    elif recording_ids:
        query = query.filter(Prediction.recording_id.in_(recording_ids))
    if filters.device_id:
        query = query.filter(Recording.device_id == filters.device_id)
    if filters.tenant_id:
        query = query.filter(Recording.tenant_id == filters.tenant_id)
    if filters.model_version:
        query = query.filter(Prediction.model_version == filters.model_version)
    if filters.feature_schema_version:
        query = query.filter(
            Prediction.feature_schema_version == filters.feature_schema_version
        )
    if filters.from_ts:
        query = query.filter(Prediction.window_end_ts >= filters.from_ts)
    if filters.to_ts:
        query = query.filter(Prediction.window_end_ts <= filters.to_ts)
    rows = query.all()
    label_lookup: dict[tuple[str, datetime], dict[str, Any]] = {}
    for recording_id, window_end_ts, predicted_stage, ground_truth_stage in rows:
        label_lookup[(str(recording_id), window_end_ts)] = {
            "predicted_stage": predicted_stage,
            "ground_truth_stage": ground_truth_stage,
        }
    return label_lookup


def _align_labels(
    windows: list[np.ndarray],
    window_meta: list[dict[str, Any]],
    labels: dict[tuple[str, datetime], dict[str, Any]],
    *,
    label_strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[str] = []
    window_end_ts: list[str] = []
    recording_ids: list[str] = []
    for tensor, meta in zip(windows, window_meta):
        recording_id = meta.get("recording_id")
        window_ts = meta.get("window_end_ts")
        if recording_id is None or not isinstance(window_ts, datetime):
            continue
        key = (str(recording_id), window_ts)
        label_info = labels.get(key)
        if not label_info:
            continue
        label = _select_label(label_info, label_strategy)
        if label is None:
            continue
        X.append(tensor)
        y.append(label)
        window_end_ts.append(window_ts.isoformat())
        recording_ids.append(str(recording_id))
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y),
        np.asarray(window_end_ts),
        np.asarray(recording_ids),
    )


def _select_label(label_info: dict[str, Any], label_strategy: str) -> str | None:
    if label_strategy == "ground_truth_only":
        return label_info.get("ground_truth_stage")
    if label_strategy == "predicted_only":
        return label_info.get("predicted_stage")
    if label_strategy == "ground_truth_or_predicted":
        return label_info.get("ground_truth_stage") or label_info.get("predicted_stage")
    raise ValueError("Unknown label strategy")


def _balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    window_end_ts: np.ndarray,
    recording_ids: np.ndarray,
    *,
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if strategy == "none":
        return X, y, window_end_ts, recording_ids
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) == 0:
        return X, y, window_end_ts, recording_ids
    target = counts.min() if strategy == "undersample" else counts.max()
    indices: list[int] = []
    for label in unique:
        label_indices = np.where(y == label)[0]
        if strategy == "undersample":
            chosen = rng.choice(label_indices, size=target, replace=False)
        elif strategy == "oversample":
            chosen = rng.choice(label_indices, size=target, replace=True)
        else:
            raise ValueError("Unknown balance strategy")
        indices.extend(chosen.tolist())
    indices = sorted(indices)
    return X[indices], y[indices], window_end_ts[indices], recording_ids[indices]


def _stratified_split_indices(
    y: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, np.ndarray]:
    from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

    indices = np.arange(len(y))
    if len(indices) == 0:
        return {
            "train": np.array([], dtype=int),
            "val": np.array([], dtype=int),
            "test": np.array([], dtype=int),
        }
    try:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_ratio,
            random_state=seed,
        )
        train_val_idx, test_idx = next(sss.split(indices, y))
        remaining_y = y[train_val_idx]
        val_denominator = train_ratio + val_ratio
        if val_denominator <= 0:
            return {
                "train": np.array([], dtype=int),
                "val": np.array([], dtype=int),
                "test": test_idx,
            }
        val_size = val_ratio / val_denominator
        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_size,
            random_state=seed,
        )
        train_idx, val_idx = next(sss_val.split(train_val_idx, remaining_y))
        return {
            "train": train_val_idx[train_idx],
            "val": train_val_idx[val_idx],
            "test": test_idx,
        }
    except ValueError:
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=test_ratio,
            random_state=seed,
        )
        train_val_idx, test_idx = next(splitter.split(indices))
        val_denominator = train_ratio + val_ratio
        if val_denominator <= 0:
            return {
                "train": np.array([], dtype=int),
                "val": np.array([], dtype=int),
                "test": test_idx,
            }
        val_size = val_ratio / val_denominator
        val_splitter = ShuffleSplit(
            n_splits=1,
            test_size=val_size,
            random_state=seed,
        )
        train_idx, val_idx = next(val_splitter.split(train_val_idx))
        return {
            "train": train_val_idx[train_idx],
            "val": train_val_idx[val_idx],
            "test": test_idx,
        }


def _build_metadata(
    config: DatasetBuildConfig,
    feature_schema: FeatureSchema,
    label_map: list[str],
    splits: dict[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "window_size": config.window_size,
        "allow_padding": config.allow_padding,
        "feature_schema_version": feature_schema.version,
        "feature_schema_path": str(config.feature_schema_path),
        "git_commit_hash": _git_commit_hash(),
        "filters": {
            "from_ts": config.filters.from_ts.isoformat()
            if config.filters.from_ts
            else None,
            "to_ts": config.filters.to_ts.isoformat() if config.filters.to_ts else None,
            "device_id": config.filters.device_id,
            "recording_id": config.filters.recording_id,
            "feature_schema_version": config.filters.feature_schema_version,
            "model_version": config.filters.model_version,
            "tenant_id": config.filters.tenant_id,
        },
        "label_strategy": config.label_strategy,
        "balance_strategy": config.balance_strategy,
        "random_seed": config.random_seed,
        "split_sizes": {k: int(v.size) for k, v in splits.items()},
        "label_map": label_map,
    }


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


def _export_dataset(
    result: DatasetBuildResult,
    config: DatasetBuildConfig,
    feature_schema: FeatureSchema,
) -> None:
    if config.export_format == "npz":
        save_npz(
            config.output_dir,
            X=result.X,
            y=result.y,
            label_map=result.label_map,
            window_end_ts=result.window_end_ts,
            recording_ids=result.recording_ids,
            splits=result.splits,
            metadata=result.metadata,
        )
        (config.output_dir / "feature_schema.json").write_text(
            json.dumps(
                {
                    "version": feature_schema.version,
                    "features": feature_schema.features,
                },
                indent=2,
            )
        )
        return
    if config.export_format == "parquet":
        export_parquet(
            config.output_dir,
            X=result.X,
            y=result.y,
            label_map=result.label_map,
            window_end_ts=result.window_end_ts,
            recording_ids=result.recording_ids,
            splits=result.splits,
            metadata=result.metadata,
            feature_names=feature_schema.features,
        )
        return
    if config.export_format == "hdf5":
        export_hdf5(
            config.output_dir,
            X=result.X,
            y=result.y,
            label_map=result.label_map,
            window_end_ts=result.window_end_ts,
            recording_ids=result.recording_ids,
            splits=result.splits,
            metadata=result.metadata,
        )
        return
    raise ValueError("Unsupported export format")
