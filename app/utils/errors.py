from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ErrorDetail:
    code: str
    message: str
    classification: str
    status_code: int
    extra: dict[str, Any] | None = None


class AppError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        classification: str,
        status_code: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.detail = ErrorDetail(
            code=code,
            message=message,
            classification=classification,
            status_code=status_code,
            extra=extra,
        )


class ModelUnavailableError(AppError):
    def __init__(self, message: str = "Model unavailable") -> None:
        super().__init__(
            code="model_unavailable",
            message=message,
            classification="dependency",
            status_code=503,
        )


class CircuitBreakerOpenError(AppError):
    def __init__(self, message: str = "Database circuit open") -> None:
        super().__init__(
            code="db_circuit_open",
            message=message,
            classification="dependency",
            status_code=503,
        )


class RequestTimeoutError(AppError):
    def __init__(self, message: str = "Request timeout") -> None:
        super().__init__(
            code="request_timeout",
            message=message,
            classification="transient",
            status_code=504,
        )
