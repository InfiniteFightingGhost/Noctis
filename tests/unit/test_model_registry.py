from __future__ import annotations

from pathlib import Path

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
