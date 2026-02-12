from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class FaultState:
    name: str
    enabled_at: datetime
    expires_at: datetime | None
    params: dict[str, Any]

    def is_active(self, now: datetime) -> bool:
        if self.expires_at is None:
            return True
        return now < self.expires_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled_at": self.enabled_at,
            "expires_at": self.expires_at,
            "params": self.params,
        }


_ALLOWED_FAULTS = {"db_down", "db_latency_ms", "model_unavailable", "timeout"}
_faults: dict[str, FaultState] = {}
_lock = RLock()


def enable_fault(
    name: str, *, ttl_seconds: int | None = None, params: dict[str, Any] | None = None
) -> FaultState:
    if name not in _ALLOWED_FAULTS:
        raise ValueError("Unsupported fault name")
    if ttl_seconds is not None and ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")
    if params is None:
        params = {}
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=ttl_seconds) if ttl_seconds else None
    state = FaultState(
        name=name,
        enabled_at=now,
        expires_at=expires_at,
        params=params,
    )
    with _lock:
        _faults[name] = state
    return state


def disable_fault(name: str) -> bool:
    with _lock:
        return _faults.pop(name, None) is not None


def list_faults() -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    with _lock:
        _expire_faults(now)
        return [state.as_dict() for state in _faults.values()]


def get_fault(name: str) -> FaultState | None:
    now = datetime.now(timezone.utc)
    with _lock:
        _expire_faults(now)
        state = _faults.get(name)
        if state and state.is_active(now):
            return state
        return None


def is_fault_active(name: str) -> bool:
    return get_fault(name) is not None


def _expire_faults(now: datetime) -> None:
    expired = [key for key, state in _faults.items() if not state.is_active(now)]
    for key in expired:
        _faults.pop(key, None)
