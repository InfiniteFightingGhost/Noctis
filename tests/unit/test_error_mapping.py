from __future__ import annotations

from app.utils.error_payloads import error_payload, failure_classification


def test_failure_classification_dependency_transient() -> None:
    assert failure_classification("db_error", "dependency") == "TRANSIENT"


def test_failure_classification_client_fatal() -> None:
    assert failure_classification("validation_error", "client") == "FATAL"


def test_error_payload_includes_failure_classification() -> None:
    payload = error_payload(
        code="request_timeout",
        message="timeout",
        classification="transient",
    )
    assert payload["error"]["failure_classification"] == "TRANSIENT"
