from __future__ import annotations

from app.db.session import DbCircuitBreaker


def test_circuit_breaker_opens() -> None:
    breaker = DbCircuitBreaker(failure_threshold=2, recovery_seconds=30)
    assert breaker.allow_request()
    breaker.record_failure()
    assert breaker.allow_request()
    breaker.record_failure()
    assert not breaker.allow_request()
    assert breaker.state == "open"
