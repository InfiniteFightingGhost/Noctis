from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.dataset.config import DatasetBuildConfig
from app.dataset.io import export_hdf5, export_parquet, save_npz
from app.db.models import Epoch, Prediction, Recording
from app.db.session import run_with_db_retry
from app.feature_store.schema import FeatureSchemaRecord
from app.feature_store.service import get_feature_schema_by_version
from app.ml.feature_decode import decode_features
from app.ml.feature_schema import load_feature_schema
from app.ml.validation import ensure_finite
from app.services.windowing import WindowedEpoch, build_windows


@dataclass(frozen=True)
class DatasetBuildResult:
    X: np.ndarray
    y: np.ndarray
    label_sources: np.ndarray
    label_map: list[str]
    window_end_ts: np.ndarray
    recording_ids: np.ndarray
    splits: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _resolve_feature_schema(session: Session, config: DatasetBuildConfig) -> FeatureSchemaRecord:
    if config.feature_schema_version:
        schema = get_feature_schema_by_version(session, config.feature_schema_version)
        if schema is None:
            raise ValueError("Feature schema not registered")
        return schema
    if config.feature_schema_path:
        file_schema = load_feature_schema(config.feature_schema_path)
        schema = get_feature_schema_by_version(session, file_schema.version)
        if schema is None:
            raise ValueError("Feature schema not registered")
        if schema.feature_names != file_schema.features:
            raise ValueError("Feature schema ordering mismatch")
        return schema
    raise ValueError("feature_schema_version is required")


def build_dataset(config: DatasetBuildConfig) -> DatasetBuildResult:
    def _op(session: Session) -> DatasetBuildResult:
        feature_schema = _resolve_feature_schema(session, config)
        label_strategy = _resolve_label_strategy(config)
        _validate_label_strategy(label_strategy, config.allow_predicted_labels)
        epochs = _fetch_epochs(session, config, feature_schema)
        windows, window_meta = _build_windows(epochs, config)
        labels = _fetch_labels(session, config, window_meta, feature_schema.version)
        X, y, label_sources, window_end_ts, recording_ids = _align_labels(
            windows,
            window_meta,
            labels,
            label_strategy=label_strategy,
        )
        if X.size == 0:
            raise ValueError("No labeled windows found for dataset")
        X, y, label_sources, window_end_ts, recording_ids = _balance_classes(
            X,
            y,
            label_sources,
            window_end_ts,
            recording_ids,
            strategy=config.balance_strategy,
            seed=config.random_seed,
        )
        label_source_counts = _label_source_counts(label_sources)
        padded_window_count = _count_padded_windows(window_meta, window_end_ts, recording_ids)
        split_purge_gap = _resolve_split_purge_gap(config)
        splits = _recording_split_indices(
            recording_ids,
            window_end_ts,
            train_ratio=config.split.train,
            val_ratio=config.split.val,
            test_ratio=config.split.test,
            seed=config.random_seed,
            split_strategy=config.split_strategy,
            split_time_aware=config.split_time_aware,
            split_purge_gap=split_purge_gap,
            split_block_seconds=config.split_block_seconds,
        )
        _validate_split_integrity(
            recording_ids,
            splits,
            enforce_recording_unique=config.split_strategy == "recording",
            allow_unassigned=split_purge_gap > 0,
        )
        label_map = sorted({label for label in y})
        metadata = _build_metadata(
            config,
            feature_schema,
            label_map,
            splits,
            label_strategy=label_strategy,
            label_source_counts=label_source_counts,
            padded_window_count=padded_window_count,
            split_purge_gap=split_purge_gap,
        )
        result = DatasetBuildResult(
            X=X,
            y=y,
            label_sources=label_sources,
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
    feature_schema: FeatureSchemaRecord,
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
    if filters.feature_schema_version and (
        filters.feature_schema_version != feature_schema.version
    ):
        raise ValueError("Feature schema filter mismatch")
    query = query.filter(Epoch.feature_schema_version == feature_schema.version)
    rows = query.order_by(Epoch.recording_id, Epoch.epoch_index, Epoch.epoch_start_ts).all()
    epochs: dict[object, list[WindowedEpoch]] = {}
    for recording_id, epoch_index, epoch_start_ts, schema_version, payload in rows:
        if schema_version != feature_schema.version:
            raise ValueError(
                f"Feature schema mismatch: expected {feature_schema.version}, got {schema_version}"
            )
        vector = decode_features(payload, feature_schema)
        ensure_finite("features", vector)
        key = recording_id
        epochs.setdefault(key, []).append(
            WindowedEpoch(
                epoch_index=epoch_index,
                epoch_start_ts=epoch_start_ts,
                features=vector,
                feature_schema_id=feature_schema.id,
            )
        )
    return epochs


def _build_windows(
    epochs: dict[object, list[WindowedEpoch]],
    config: DatasetBuildConfig,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    if not epochs:
        return [], []
    allow_padding = _allow_padding(config)
    tensors: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for recording_id, items in sorted(epochs.items(), key=lambda item: str(item[0])):
        windows = build_windows(
            items,
            window_size=config.window_size,
            allow_padding=allow_padding,
            epoch_seconds=config.epoch_seconds,
        )
        for window in windows:
            tensors.append(window.tensor)
            meta.append(
                {
                    "recording_id": recording_id,
                    "window_end_ts": window.end_ts,
                    "window_start_ts": window.start_ts,
                    "window_start_index": window.start_index,
                    "window_end_index": window.end_index,
                    "window_padded": window.padded,
                }
            )
    return tensors, meta


def _fetch_labels(
    session: Session,
    config: DatasetBuildConfig,
    window_meta: list[dict[str, Any]],
    feature_schema_version: str,
) -> dict[tuple[str, datetime], dict[str, Any]]:
    filters = config.filters
    recording_ids = {meta.get("recording_id") for meta in window_meta if meta.get("recording_id")}
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
    if filters.feature_schema_version and (
        filters.feature_schema_version != feature_schema_version
    ):
        raise ValueError("Feature schema filter mismatch")
    query = query.filter(Prediction.feature_schema_version == feature_schema_version)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[str] = []
    label_sources: list[str] = []
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
        label, source = _select_label(label_info, label_strategy)
        if label is None or source is None:
            continue
        X.append(tensor)
        y.append(label)
        label_sources.append(source)
        window_end_ts.append(window_ts.isoformat())
        recording_ids.append(str(recording_id))
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y),
        np.asarray(label_sources),
        np.asarray(window_end_ts),
        np.asarray(recording_ids),
    )


def _select_label(label_info: dict[str, Any], label_strategy: str) -> tuple[str | None, str | None]:
    if label_strategy == "ground_truth_only":
        label = label_info.get("ground_truth_stage")
        return label, "ground_truth" if label else None
    if label_strategy == "predicted_only":
        label = label_info.get("predicted_stage")
        return label, "predicted" if label else None
    if label_strategy == "ground_truth_or_predicted":
        ground_truth = label_info.get("ground_truth_stage")
        if ground_truth:
            return ground_truth, "ground_truth"
        predicted = label_info.get("predicted_stage")
        return predicted, "predicted" if predicted else None
    raise ValueError("Unknown label strategy")


def _resolve_label_strategy(config: DatasetBuildConfig) -> str:
    strategy = config.label_source_policy or config.label_strategy
    return str(strategy)


def _validate_label_strategy(label_strategy: str, allow_predicted_labels: bool) -> None:
    if label_strategy == "ground_truth_only":
        return
    if not allow_predicted_labels:
        raise ValueError("Predicted labels are disabled for dataset build")


def _allow_padding(config: DatasetBuildConfig) -> bool:
    if config.padding_policy:
        return config.padding_policy == "zero_fill"
    return config.allow_padding


def _label_source_counts(label_sources: np.ndarray) -> dict[str, int]:
    if label_sources.size == 0:
        return {}
    unique, counts = np.unique(label_sources, return_counts=True)
    return {str(label): int(count) for label, count in zip(unique, counts)}


def _count_padded_windows(
    window_meta: list[dict[str, Any]],
    window_end_ts: np.ndarray,
    recording_ids: np.ndarray,
) -> int:
    lookup: dict[tuple[str, str], bool] = {}
    for meta in window_meta:
        recording_id = meta.get("recording_id")
        window_ts = meta.get("window_end_ts")
        if recording_id is None or not isinstance(window_ts, datetime):
            continue
        lookup[(str(recording_id), window_ts.isoformat())] = bool(meta.get("window_padded"))
    count = 0
    for recording_id, window_ts in zip(recording_ids, window_end_ts, strict=False):
        if lookup.get((str(recording_id), str(window_ts))):
            count += 1
    return count


def _resolve_split_purge_gap(config: DatasetBuildConfig) -> int:
    if config.split_purge_gap > 0:
        return config.split_purge_gap
    if config.split_time_aware or config.split_strategy == "recording_time":
        return max(config.window_size - 1, 0)
    return 0


def _balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    label_sources: np.ndarray,
    window_end_ts: np.ndarray,
    recording_ids: np.ndarray,
    *,
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if strategy == "none":
        return X, y, label_sources, window_end_ts, recording_ids
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) == 0:
        return X, y, label_sources, window_end_ts, recording_ids
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
    return (
        X[indices],
        y[indices],
        label_sources[indices],
        window_end_ts[indices],
        recording_ids[indices],
    )


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


def _recording_split_indices(
    recording_ids: np.ndarray,
    window_end_ts: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_strategy: str,
    split_time_aware: bool,
    split_purge_gap: int,
    split_block_seconds: int | None,
) -> dict[str, np.ndarray]:
    if recording_ids.size == 0:
        return {
            "train": np.array([], dtype=int),
            "val": np.array([], dtype=int),
            "test": np.array([], dtype=int),
        }
    groups = _build_split_groups(
        recording_ids,
        window_end_ts,
        split_strategy=split_strategy,
        split_time_aware=split_time_aware,
        split_block_seconds=split_block_seconds,
        seed=seed,
    )
    total = len(groups)
    test_count = int(round(total * test_ratio))
    val_count = int(round(total * val_ratio))
    if test_count + val_count > total:
        test_count = min(test_count, total)
        val_count = max(0, total - test_count)
    train_count = max(0, total - test_count - val_count)
    train_cutoff = train_count
    val_cutoff = train_count + val_count
    assigned = 0
    split_assignments: dict[str, list[int]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for group in groups:
        if assigned < train_cutoff:
            target = "train"
        elif assigned < val_cutoff:
            target = "val"
        else:
            target = "test"
        split_assignments[target].extend(group["indices"])
        assigned += 1
    splits = {
        name: np.asarray(sorted(indices), dtype=int) for name, indices in split_assignments.items()
    }
    if split_purge_gap > 0:
        splits = _apply_split_purge_gap(
            recording_ids=recording_ids,
            window_end_ts=window_end_ts,
            splits=splits,
            purge_gap=split_purge_gap,
        )
    return splits


def _build_time_groups(
    recording_ids: np.ndarray,
    window_end_ts: np.ndarray,
) -> list[set[str]]:
    ranges: dict[str, tuple[datetime, datetime]] = {}
    for recording_id, window_ts in zip(recording_ids, window_end_ts, strict=False):
        ts = _coerce_timestamp(window_ts)
        if ts is None:
            continue
        key = str(recording_id)
        existing = ranges.get(key)
        if existing is None:
            ranges[key] = (ts, ts)
        else:
            start, end = existing
            ranges[key] = (min(start, ts), max(end, ts))
    ordered = sorted(ranges.items(), key=lambda item: item[1][0])
    groups: list[set[str]] = []
    current_group: set[str] = set()
    current_start: datetime | None = None
    current_end: datetime | None = None
    for recording_id, (start, end) in ordered:
        if current_start is None:
            current_group = {recording_id}
            current_start = start
            current_end = end
            continue
        if start <= (current_end or start):
            current_group.add(recording_id)
            current_end = max(current_end or end, end)
            continue
        groups.append(set(current_group))
        current_group = {recording_id}
        current_start = start
        current_end = end
    if current_group:
        groups.append(set(current_group))
    missing = {str(recording_id) for recording_id in recording_ids} - {
        rec for group in groups for rec in group
    }
    for rec in sorted(missing):
        groups.append({rec})
    return groups


def _build_split_groups(
    recording_ids: np.ndarray,
    window_end_ts: np.ndarray,
    *,
    split_strategy: str,
    split_time_aware: bool,
    split_block_seconds: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    records = _collect_record_windows(recording_ids, window_end_ts)
    groups: list[dict[str, Any]] = []
    if split_strategy == "recording_time":
        recording_groups = _build_time_groups(recording_ids, window_end_ts)
        for group in recording_groups:
            grouped = _group_indices_for_recordings(group, records)
            groups.extend(_split_groups_by_time(grouped, split_block_seconds))
    elif split_strategy == "recording":
        for rec_id in sorted(records):
            groups.extend(_split_groups_by_time(records[rec_id], split_block_seconds))
    else:
        raise ValueError("Unsupported split strategy")
    if split_time_aware or split_strategy == "recording_time":
        groups.sort(key=lambda item: item["start_ts"])
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
    return groups


def _collect_record_windows(
    recording_ids: np.ndarray,
    window_end_ts: np.ndarray,
) -> dict[str, list[tuple[int, datetime]]]:
    records: dict[str, list[tuple[int, datetime]]] = {}
    for idx, (recording_id, window_ts) in enumerate(
        zip(recording_ids, window_end_ts, strict=False)
    ):
        ts = _coerce_timestamp(window_ts) or datetime.min
        key = str(recording_id)
        records.setdefault(key, []).append((idx, ts))
    for key in records:
        records[key].sort(key=lambda item: (item[1], item[0]))
    return records


def _group_indices_for_recordings(
    group: set[str],
    records: dict[str, list[tuple[int, datetime]]],
) -> list[tuple[int, datetime]]:
    combined: list[tuple[int, datetime]] = []
    for rec_id in sorted(group):
        combined.extend(records.get(rec_id, []))
    combined.sort(key=lambda item: (item[1], item[0]))
    return combined


def _split_groups_by_time(
    indices: list[tuple[int, datetime]],
    split_block_seconds: int | None,
) -> list[dict[str, Any]]:
    if not indices:
        return []
    block_seconds = int(split_block_seconds or 0)
    if block_seconds <= 0:
        start_ts = indices[0][1]
        return [{"indices": [idx for idx, _ in indices], "start_ts": start_ts}]
    groups: list[dict[str, Any]] = []
    current: list[int] = []
    block_start = indices[0][1]
    for idx, ts in indices:
        if ts - block_start > timedelta(seconds=block_seconds):
            groups.append({"indices": current, "start_ts": block_start})
            current = []
            block_start = ts
        current.append(idx)
    if current:
        groups.append({"indices": current, "start_ts": block_start})
    return groups


def _apply_split_purge_gap(
    *,
    recording_ids: np.ndarray,
    window_end_ts: np.ndarray,
    splits: dict[str, np.ndarray],
    purge_gap: int,
) -> dict[str, np.ndarray]:
    if purge_gap <= 0:
        return splits
    split_labels = np.full(recording_ids.shape[0], "", dtype=object)
    for name, indices in splits.items():
        for idx in indices:
            split_labels[idx] = name
    to_drop: set[int] = set()
    for recording_id in sorted({str(rec_id) for rec_id in recording_ids}):
        indices = [idx for idx, rec_id in enumerate(recording_ids) if str(rec_id) == recording_id]
        if len(indices) < 2:
            continue
        indices.sort(
            key=lambda idx: (
                _coerce_timestamp(window_end_ts[idx]) or datetime.min,
                idx,
            )
        )
        for pos in range(1, len(indices)):
            prev_idx = indices[pos - 1]
            curr_idx = indices[pos]
            if split_labels[prev_idx] and split_labels[curr_idx]:
                if split_labels[prev_idx] != split_labels[curr_idx]:
                    start = max(0, pos - purge_gap)
                    end = min(len(indices), pos + purge_gap)
                    to_drop.update(indices[start:end])
    if not to_drop:
        return splits
    return {
        name: np.asarray([idx for idx in indices if idx not in to_drop], dtype=int)
        for name, indices in splits.items()
    }


def _coerce_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _validate_split_integrity(
    recording_ids: np.ndarray,
    splits: dict[str, np.ndarray],
    *,
    enforce_recording_unique: bool,
    allow_unassigned: bool,
) -> None:
    assigned_indices: set[int] = set()
    assigned_recordings: dict[str, str] = {}
    for split_name, indices in splits.items():
        for idx in indices:
            if idx in assigned_indices:
                raise ValueError("Window appears in multiple splits")
            assigned_indices.add(idx)
            if enforce_recording_unique:
                recording_id = str(recording_ids[idx])
                existing = assigned_recordings.get(recording_id)
                if existing and existing != split_name:
                    raise ValueError("Recording appears in multiple splits")
                assigned_recordings[recording_id] = split_name
    if not allow_unassigned:
        expected = set(range(len(recording_ids)))
        if assigned_indices != expected:
            raise ValueError("Split assignment missing windows")


def _build_metadata(
    config: DatasetBuildConfig,
    feature_schema: FeatureSchemaRecord,
    label_map: list[str],
    splits: dict[str, np.ndarray],
    *,
    label_strategy: str,
    label_source_counts: dict[str, int],
    padded_window_count: int,
    split_purge_gap: int,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "window_size": config.window_size,
        "epoch_seconds": config.epoch_seconds,
        "allow_padding": config.allow_padding,
        "padding_policy": config.padding_policy,
        "feature_schema_version": feature_schema.version,
        "feature_schema_id": str(feature_schema.id),
        "feature_schema_hash": feature_schema.hash,
        "feature_names": feature_schema.feature_names,
        "feature_schema_path": str(config.feature_schema_path)
        if config.feature_schema_path
        else None,
        "git_commit_hash": _git_commit_hash(),
        "filters": {
            "from_ts": config.filters.from_ts.isoformat() if config.filters.from_ts else None,
            "to_ts": config.filters.to_ts.isoformat() if config.filters.to_ts else None,
            "device_id": config.filters.device_id,
            "recording_id": config.filters.recording_id,
            "feature_schema_version": config.filters.feature_schema_version,
            "model_version": config.filters.model_version,
            "tenant_id": config.filters.tenant_id,
        },
        "label_strategy": label_strategy,
        "label_source_policy": label_strategy,
        "allow_predicted_labels": config.allow_predicted_labels,
        "label_source_counts": label_source_counts,
        "padded_window_count": padded_window_count,
        "balance_strategy": config.balance_strategy,
        "random_seed": config.random_seed,
        "split_sizes": {k: int(v.size) for k, v in splits.items()},
        "label_map": label_map,
        "split_policy": {
            "split_strategy": config.split_strategy,
            "seed": config.random_seed,
            "grouping_key": "recording_id",
            "time_aware": config.split_time_aware or config.split_strategy == "recording_time",
            "purge_gap": split_purge_gap,
            "block_seconds": config.split_block_seconds,
        },
        "window_alignment": {
            "end_ts": "epoch_start_plus_duration",
            "epoch_seconds": config.epoch_seconds,
            "padding_policy": config.padding_policy,
            "alignment": config.window_alignment,
        },
        "window_stride": 1,
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
    feature_schema: FeatureSchemaRecord,
) -> None:
    feature_payload = [
        {
            "name": feature.name,
            "dtype": feature.dtype,
            "allowed_range": feature.allowed_range,
            "description": feature.description,
            "introduced_in_version": feature.introduced_in_version,
            "deprecated_in_version": feature.deprecated_in_version,
            "position": feature.position,
        }
        for feature in feature_schema.features
    ]
    feature_schema_payload = {
        "id": str(feature_schema.id),
        "version": feature_schema.version,
        "hash": feature_schema.hash,
        "features": feature_payload,
    }
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
            json.dumps(feature_schema_payload, indent=2)
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
            feature_names=feature_schema.feature_names,
        )
        (config.output_dir / "feature_schema.json").write_text(
            json.dumps(feature_schema_payload, indent=2)
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
        (config.output_dir / "feature_schema.json").write_text(
            json.dumps(feature_schema_payload, indent=2)
        )
        return
    raise ValueError("Unsupported export format")
