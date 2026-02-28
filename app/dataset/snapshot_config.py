from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.dataset.config import DatasetFilters, DatasetSplitConfig, _parse_datetime


@dataclass(frozen=True)
class DatasetSnapshotConfig:
    name: str
    output_dir: Path
    feature_schema_version: str
    window_size: int = 21
    epoch_seconds: int = 30
    allow_padding: bool = False
    window_alignment: str = "epoch_end"
    padding_policy: str = "reject"
    label_source_policy: str = "ground_truth_only"
    label_strategy: str = "ground_truth_only"
    allow_predicted_labels: bool = False
    balance_strategy: str = "none"
    random_seed: int = 42
    export_format: str = "npz"
    split: DatasetSplitConfig = DatasetSplitConfig()
    filters: DatasetFilters = DatasetFilters()
    split_strategy: str = "recording"
    split_time_aware: bool = False
    split_purge_gap: int = 0
    split_block_seconds: int | None = None
    split_grouped_stratification: bool = False
    split_stratify_key: str = "dataset_rem_bucket"


def load_snapshot_config(path: Path) -> DatasetSnapshotConfig:
    payload = _load_payload(path)
    return snapshot_config_from_payload(payload)


def snapshot_config_from_payload(payload: dict[str, Any]) -> DatasetSnapshotConfig:
    filters = payload.get("filters", {}) if isinstance(payload, dict) else {}
    split_payload = payload.get("split", {}) if isinstance(payload, dict) else {}
    split = DatasetSplitConfig(
        train=float(split_payload.get("train", 0.75)),
        val=float(split_payload.get("val", 0.1)),
        test=float(split_payload.get("test", 0.15)),
    )
    split.validate()
    name = payload.get("name")
    if not name:
        raise ValueError("Snapshot name is required")
    feature_schema_version = payload.get("feature_schema_version")
    if not feature_schema_version:
        raise ValueError("feature_schema_version is required")
    split_strategy = str(payload.get("split_strategy", "recording"))
    padding_policy = payload.get("padding_policy")
    allow_padding = bool(payload.get("allow_padding", False))
    if padding_policy is None:
        padding_policy = "zero_fill" if allow_padding else "reject"
    else:
        allow_padding = str(padding_policy) == "zero_fill"
    label_source_policy = payload.get("label_source_policy") or payload.get(
        "label_strategy", "ground_truth_only"
    )
    config = DatasetSnapshotConfig(
        name=str(name),
        output_dir=Path(payload["output_dir"]),
        feature_schema_version=str(feature_schema_version),
        window_size=int(payload.get("window_size", 21)),
        epoch_seconds=int(payload.get("epoch_seconds", 30)),
        allow_padding=allow_padding,
        window_alignment=str(payload.get("window_alignment", "epoch_end")),
        padding_policy=str(padding_policy),
        label_source_policy=str(label_source_policy),
        label_strategy=str(payload.get("label_strategy", label_source_policy)),
        allow_predicted_labels=bool(payload.get("allow_predicted_labels", False)),
        balance_strategy=str(payload.get("balance_strategy", "none")),
        random_seed=int(payload.get("random_seed", 42)),
        export_format=str(payload.get("export_format", "npz")),
        split=split,
        filters=DatasetFilters(
            from_ts=_parse_datetime(filters.get("from_ts")),
            to_ts=_parse_datetime(filters.get("to_ts")),
            device_id=filters.get("device_id"),
            recording_id=filters.get("recording_id"),
            feature_schema_version=filters.get("feature_schema_version"),
            model_version=filters.get("model_version"),
            tenant_id=filters.get("tenant_id"),
        ),
        split_strategy=split_strategy,
        split_time_aware=bool(payload.get("split_time_aware", split_strategy == "recording_time")),
        split_purge_gap=int(payload.get("split_purge_gap", 0)),
        split_block_seconds=(
            int(payload["split_block_seconds"])
            if payload.get("split_block_seconds") is not None
            else None
        ),
        split_grouped_stratification=bool(payload.get("split_grouped_stratification", False)),
        split_stratify_key=str(payload.get("split_stratify_key", "dataset_rem_bucket")),
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
