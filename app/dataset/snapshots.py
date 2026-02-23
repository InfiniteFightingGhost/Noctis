from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.dataset.builder import (
    DatasetBuildResult,
    _stratified_split_indices,
    build_dataset,
)
from app.dataset.config import DatasetBuildConfig, DatasetSplitConfig
from app.dataset.io import export_hdf5, export_parquet, save_npz
from app.dataset.snapshot_config import DatasetSnapshotConfig
from app.db.models import DatasetSnapshot, DatasetSnapshotWindow, Epoch
from app.db.session import run_with_db_retry
from app.feature_store.service import get_feature_schema_by_version
from app.ml.feature_decode import decode_features
from app.reproducibility.snapshots import compute_snapshot_checksum, snapshot_window_id
from app.services.windowing import WindowedEpoch, build_windows


@dataclass(frozen=True)
class SnapshotCreateResult:
    snapshot_id: uuid.UUID
    checksum: str
    row_count: int
    output_dir: Path


def create_snapshot(config: DatasetSnapshotConfig) -> SnapshotCreateResult:
    build_config = DatasetBuildConfig(
        output_dir=config.output_dir,
        feature_schema_path=None,
        feature_schema_version=config.feature_schema_version,
        window_size=config.window_size,
        allow_padding=config.allow_padding,
        label_strategy=config.label_strategy,
        balance_strategy=config.balance_strategy,
        random_seed=config.random_seed,
        export_format=config.export_format,
        split=config.split,
        filters=config.filters,
    )
    result = build_dataset(build_config)
    entries = _snapshot_entries(result)
    checksum = compute_snapshot_checksum(entries)
    row_count = int(result.y.size)
    date_range_start, date_range_end = _snapshot_date_range(result)
    recording_filter = _recording_filter_payload(config)

    def _insert(session: Session) -> uuid.UUID:
        snapshot = DatasetSnapshot(
            name=config.name,
            feature_schema_version=config.feature_schema_version,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            recording_filter=recording_filter,
            label_source=config.label_strategy,
            created_at=datetime.now(timezone.utc),
            checksum=checksum,
            row_count=row_count,
        )
        session.add(snapshot)
        session.flush()
        windows = []
        for idx, (recording_id, window_ts, label, label_source) in enumerate(
            _snapshot_rows(result),
            start=0,
        ):
            windows.append(
                DatasetSnapshotWindow(
                    dataset_snapshot_id=snapshot.id,
                    window_order=idx,
                    recording_id=uuid.UUID(recording_id),
                    window_end_ts=datetime.fromisoformat(window_ts),
                    label_value=label,
                    label_source=label_source,
                )
            )
        session.add_all(windows)
        return snapshot.id

    snapshot_id = run_with_db_retry(
        _insert,
        commit=True,
        operation_name="dataset_snapshot_create",
    )
    _update_snapshot_metadata(
        config.output_dir,
        snapshot_id=snapshot_id,
        checksum=checksum,
        row_count=row_count,
        recording_filter=recording_filter,
    )
    return SnapshotCreateResult(
        snapshot_id=snapshot_id,
        checksum=checksum,
        row_count=row_count,
        output_dir=config.output_dir,
    )


def reload_snapshot(
    *,
    snapshot_id: uuid.UUID,
    output_dir: Path | None = None,
    export_format: str | None = None,
) -> DatasetBuildResult:
    result = run_with_db_retry(
        lambda session: _load_snapshot(session, snapshot_id),
        operation_name="snapshot_reload",
    )
    if output_dir:
        export_format = export_format or "npz"
        _export_snapshot(result, output_dir, export_format)
    return result


def _load_snapshot(session: Session, snapshot_id: uuid.UUID) -> DatasetBuildResult:
    snapshot = session.execute(
        select(DatasetSnapshot).where(DatasetSnapshot.id == snapshot_id)
    ).scalar_one_or_none()
    if snapshot is None:
        raise ValueError("Dataset snapshot not found")
    schema = get_feature_schema_by_version(session, snapshot.feature_schema_version)
    if schema is None:
        raise ValueError("Feature schema not registered")
    windows = (
        session.execute(
            select(DatasetSnapshotWindow)
            .where(DatasetSnapshotWindow.dataset_snapshot_id == snapshot_id)
            .order_by(DatasetSnapshotWindow.window_order)
        )
        .scalars()
        .all()
    )
    if not windows:
        raise ValueError("Snapshot contains no windows")
    recording_ids = {window.recording_id for window in windows}
    epochs = _fetch_epochs(session, recording_ids, schema)
    window_size = int((snapshot.recording_filter or {}).get("window_size") or 21)
    allow_padding = bool(
        (snapshot.recording_filter or {}).get("allow_padding") or False
    )
    window_map = _build_window_map(epochs, window_size, allow_padding)
    X: list[np.ndarray] = []
    y: list[str] = []
    label_sources: list[str] = []
    window_end_ts: list[str] = []
    recording_list: list[str] = []
    for window in windows:
        tensors = window_map.get(window.recording_id, {})
        tensor = tensors.get(window.window_end_ts)
        if tensor is None:
            raise ValueError("Snapshot window not found in epochs")
        X.append(tensor)
        y.append(str(window.label_value))
        label_sources.append(window.label_source or "unknown")
        window_end_ts.append(window.window_end_ts.isoformat())
        recording_list.append(str(window.recording_id))
    split_config = _split_config(snapshot.recording_filter)
    splits = _stratified_split_indices(
        np.asarray(y),
        train_ratio=split_config.train,
        val_ratio=split_config.val,
        test_ratio=split_config.test,
        seed=_random_seed(snapshot.recording_filter),
    )
    label_map = sorted({label for label in y})
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_snapshot_id": str(snapshot.id),
        "dataset_snapshot_checksum": snapshot.checksum,
        "dataset_snapshot_row_count": snapshot.row_count,
        "feature_schema_version": snapshot.feature_schema_version,
        "feature_names": schema.feature_names,
        "label_strategy": snapshot.label_source,
        "filters": snapshot.recording_filter or {},
        "split_sizes": {k: int(v.size) for k, v in splits.items()},
        "label_map": label_map,
    }
    return DatasetBuildResult(
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y),
        label_sources=np.asarray(label_sources),
        label_map=label_map,
        window_end_ts=np.asarray(window_end_ts),
        recording_ids=np.asarray(recording_list),
        splits=splits,
        metadata=metadata,
    )


def _fetch_epochs(
    session: Session,
    recording_ids: set[uuid.UUID],
    schema,
) -> dict[uuid.UUID, list[WindowedEpoch]]:
    rows = (
        session.query(
            Epoch.recording_id,
            Epoch.epoch_index,
            Epoch.epoch_start_ts,
            Epoch.feature_schema_version,
            Epoch.features_payload,
        )
        .filter(Epoch.recording_id.in_(recording_ids))
        .filter(Epoch.feature_schema_version == schema.version)
        .order_by(Epoch.recording_id, Epoch.epoch_index)
        .all()
    )
    epochs: dict[uuid.UUID, list[WindowedEpoch]] = {}
    for recording_id, epoch_index, epoch_start_ts, schema_version, payload in rows:
        if schema_version != schema.version:
            raise ValueError("Feature schema mismatch for snapshot reload")
        vector = decode_features(payload, schema)
        epochs.setdefault(recording_id, []).append(
            WindowedEpoch(
                epoch_index=epoch_index,
                epoch_start_ts=epoch_start_ts,
                features=vector,
                feature_schema_id=schema.id,
            )
        )
    return epochs


def _build_window_map(
    epochs: dict[uuid.UUID, list[WindowedEpoch]],
    window_size: int,
    allow_padding: bool,
) -> dict[uuid.UUID, dict[datetime, np.ndarray]]:
    window_map: dict[uuid.UUID, dict[datetime, np.ndarray]] = {}
    for recording_id, items in epochs.items():
        windows = build_windows(
            items, window_size=window_size, allow_padding=allow_padding
        )
        per_recording: dict[datetime, np.ndarray] = {}
        for window in windows:
            per_recording[window.end_ts] = window.tensor
        window_map[recording_id] = per_recording
    return window_map


def _snapshot_entries(result: DatasetBuildResult) -> list[tuple[str, str | None]]:
    entries: list[tuple[str, str | None]] = []
    for recording_id, window_ts, label in zip(
        result.recording_ids, result.window_end_ts, result.y, strict=True
    ):
        window_id = snapshot_window_id(
            uuid.UUID(str(recording_id)), datetime.fromisoformat(str(window_ts))
        )
        entries.append((window_id, str(label)))
    return entries


def _snapshot_rows(
    result: DatasetBuildResult,
) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for recording_id, window_ts, label, source in zip(
        result.recording_ids,
        result.window_end_ts,
        result.y,
        result.label_sources,
        strict=True,
    ):
        rows.append((str(recording_id), str(window_ts), str(label), str(source)))
    return rows


def _snapshot_date_range(result: DatasetBuildResult) -> tuple[datetime, datetime]:
    if result.window_end_ts.size == 0:
        raise ValueError("Snapshot has no windows")
    timestamps = [datetime.fromisoformat(str(ts)) for ts in result.window_end_ts]
    return min(timestamps), max(timestamps)


def _recording_filter_payload(config: DatasetSnapshotConfig) -> dict[str, Any]:
    return {
        "from_ts": config.filters.from_ts.isoformat()
        if config.filters.from_ts
        else None,
        "to_ts": config.filters.to_ts.isoformat() if config.filters.to_ts else None,
        "device_id": config.filters.device_id,
        "recording_id": config.filters.recording_id,
        "feature_schema_version": config.feature_schema_version,
        "model_version": config.filters.model_version,
        "tenant_id": config.filters.tenant_id,
        "window_size": config.window_size,
        "allow_padding": config.allow_padding,
        "label_strategy": config.label_strategy,
        "balance_strategy": config.balance_strategy,
        "random_seed": config.random_seed,
        "split": {
            "train": config.split.train,
            "val": config.split.val,
            "test": config.split.test,
        },
    }


def _update_snapshot_metadata(
    output_dir: Path,
    *,
    snapshot_id: uuid.UUID,
    checksum: str,
    row_count: int,
    recording_filter: dict[str, Any],
) -> None:
    path = output_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if path.exists():
        metadata = json.loads(path.read_text())
    metadata.update(
        {
            "dataset_snapshot_id": str(snapshot_id),
            "dataset_snapshot_checksum": checksum,
            "dataset_snapshot_row_count": row_count,
            "recording_filter": recording_filter,
        }
    )
    path.write_text(json.dumps(metadata, indent=2))


def _export_snapshot(
    result: DatasetBuildResult, output_dir: Path, export_format: str
) -> None:
    if export_format == "npz":
        save_npz(
            output_dir,
            X=result.X,
            y=result.y,
            label_map=result.label_map,
            window_end_ts=result.window_end_ts,
            recording_ids=result.recording_ids,
            splits=result.splits,
            metadata=result.metadata,
        )
        return
    if export_format == "parquet":
        feature_names = result.metadata.get("feature_names") or []
        export_parquet(
            output_dir,
            X=result.X,
            y=result.y,
            label_map=result.label_map,
            window_end_ts=result.window_end_ts,
            recording_ids=result.recording_ids,
            splits=result.splits,
            metadata=result.metadata,
            feature_names=feature_names,
        )
        return
    if export_format == "hdf5":
        export_hdf5(
            output_dir,
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


def _split_config(recording_filter: dict[str, Any] | None) -> DatasetSplitConfig:
    filters = recording_filter or {}
    split = filters.get("split") or {}
    config = DatasetSplitConfig(
        train=float(split.get("train", 0.7)),
        val=float(split.get("val", 0.15)),
        test=float(split.get("test", 0.15)),
    )
    config.validate()
    return config


def _random_seed(recording_filter: dict[str, Any] | None) -> int:
    filters = recording_filter or {}
    return int(filters.get("random_seed", 42))
