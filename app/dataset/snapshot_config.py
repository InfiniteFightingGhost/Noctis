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
    allow_padding: bool = False
    label_strategy: str = "ground_truth_or_predicted"
    balance_strategy: str = "none"
    random_seed: int = 42
    export_format: str = "npz"
    split: DatasetSplitConfig = DatasetSplitConfig()
    filters: DatasetFilters = DatasetFilters()


def load_snapshot_config(path: Path) -> DatasetSnapshotConfig:
    payload = _load_payload(path)
    return snapshot_config_from_payload(payload)


def snapshot_config_from_payload(payload: dict[str, Any]) -> DatasetSnapshotConfig:
    filters = payload.get("filters", {}) if isinstance(payload, dict) else {}
    split_payload = payload.get("split", {}) if isinstance(payload, dict) else {}
    split = DatasetSplitConfig(
        train=float(split_payload.get("train", 0.7)),
        val=float(split_payload.get("val", 0.15)),
        test=float(split_payload.get("test", 0.15)),
    )
    split.validate()
    name = payload.get("name")
    if not name:
        raise ValueError("Snapshot name is required")
    feature_schema_version = payload.get("feature_schema_version")
    if not feature_schema_version:
        raise ValueError("feature_schema_version is required")
    config = DatasetSnapshotConfig(
        name=str(name),
        output_dir=Path(payload["output_dir"]),
        feature_schema_version=str(feature_schema_version),
        window_size=int(payload.get("window_size", 21)),
        allow_padding=bool(payload.get("allow_padding", False)),
        label_strategy=str(payload.get("label_strategy", "ground_truth_or_predicted")),
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
