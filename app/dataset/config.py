from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSplitConfig:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if total <= 0:
            raise ValueError("Split ratios must sum to > 0")
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")


@dataclass(frozen=True)
class DatasetFilters:
    from_ts: datetime | None = None
    to_ts: datetime | None = None
    device_id: str | None = None
    recording_id: str | None = None
    feature_schema_version: str | None = None
    model_version: str | None = None
    tenant_id: str | None = None


@dataclass(frozen=True)
class DatasetBuildConfig:
    output_dir: Path
    feature_schema_path: Path
    window_size: int = 21
    allow_padding: bool = False
    label_strategy: str = "ground_truth_or_predicted"
    balance_strategy: str = "none"
    random_seed: int = 42
    export_format: str = "npz"
    split: DatasetSplitConfig = DatasetSplitConfig()
    filters: DatasetFilters = DatasetFilters()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def load_dataset_config(path: Path) -> DatasetBuildConfig:
    payload = _load_payload(path)
    return dataset_config_from_payload(payload)


def dataset_config_from_payload(payload: dict[str, Any]) -> DatasetBuildConfig:
    filters = payload.get("filters", {}) if isinstance(payload, dict) else {}
    split_payload = payload.get("split", {}) if isinstance(payload, dict) else {}
    split = DatasetSplitConfig(
        train=float(split_payload.get("train", 0.7)),
        val=float(split_payload.get("val", 0.15)),
        test=float(split_payload.get("test", 0.15)),
    )
    split.validate()
    config = DatasetBuildConfig(
        output_dir=Path(payload["output_dir"]),
        feature_schema_path=Path(payload["feature_schema_path"]),
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
