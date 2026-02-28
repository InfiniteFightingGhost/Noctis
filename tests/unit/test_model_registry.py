from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ml.registry import ModelRegistry
from app.utils.errors import ModelUnavailableError


def test_model_registry_load_and_cache() -> None:
    registry = ModelRegistry(Path("models"), "active")
    loaded = registry.get_loaded()
    cached = registry.get_loaded()
    assert loaded is cached
    assert loaded.version == "active"


def test_model_registry_reload_refreshes() -> None:
    registry = ModelRegistry(Path("models"), "active")
    loaded = registry.get_loaded()
    reloaded = registry.reload()
    assert reloaded.version == "active"
    assert reloaded is not loaded


def test_model_registry_records_last_error(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path, "missing")
    with pytest.raises(ModelUnavailableError):
        registry.load_active()
    assert registry.last_error is not None


def test_model_registry_sequence_engineered_dim_validation(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "v1"
    model_dir.mkdir(parents=True)

    bundle = SimpleNamespace(
        model=SimpleNamespace(n_features_in_=None),
        metadata={
            "feature_strategy": "sequence",
            "expected_input_dim": 26,
            "base_input_dim": 15,
            "feature_pipeline": {
                "final_feature_schema": [f"f{i}" for i in range(26)],
            },
        },
    )
    schema = SimpleNamespace(size=15, version="v1", schema_hash="abc")
    monkeypatch.setattr("app.ml.registry.load_model", lambda _path: bundle)
    monkeypatch.setattr("app.ml.registry.load_feature_schema", lambda _path: schema)
    registry = ModelRegistry(tmp_path, "v1")
    loaded = registry.load_active()
    assert loaded.metadata["expected_input_dim"] == 26


def test_model_registry_sequence_base_dim_mismatch_raises(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "v2"
    model_dir.mkdir(parents=True)
    bundle = SimpleNamespace(
        model=SimpleNamespace(n_features_in_=None),
        metadata={
            "feature_strategy": "sequence",
            "expected_input_dim": 26,
            "base_input_dim": 14,
            "feature_pipeline": {"final_feature_schema": [f"f{i}" for i in range(26)]},
        },
    )
    schema = SimpleNamespace(size=15, version="v1", schema_hash="abc")
    monkeypatch.setattr("app.ml.registry.load_model", lambda _path: bundle)
    monkeypatch.setattr("app.ml.registry.load_feature_schema", lambda _path: schema)
    registry = ModelRegistry(tmp_path, "v2")
    with pytest.raises(ModelUnavailableError, match="base feature size mismatch"):
        registry.load_active()
