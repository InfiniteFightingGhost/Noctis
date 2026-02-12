from __future__ import annotations

from app.resilience.faults import (
    disable_fault,
    enable_fault,
    get_fault,
    is_fault_active,
    list_faults,
)


def test_fault_enable_disable() -> None:
    state = enable_fault("model_unavailable", ttl_seconds=60, params={"reason": "test"})
    assert state.name == "model_unavailable"
    assert is_fault_active("model_unavailable")
    assert get_fault("model_unavailable") is not None
    faults = list_faults()
    assert any(item["name"] == "model_unavailable" for item in faults)
    assert disable_fault("model_unavailable") is True
    assert not is_fault_active("model_unavailable")


def test_fault_ttl_metadata() -> None:
    state = enable_fault("db_latency_ms", ttl_seconds=30, params={"latency_ms": 10})
    assert state.expires_at is not None
    assert state.params["latency_ms"] == 10
    disable_fault("db_latency_ms")
