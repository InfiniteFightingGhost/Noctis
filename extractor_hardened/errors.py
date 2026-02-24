from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def to_failure_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ExtractionError):
        return {
            "error_code": exc.code,
            "message": exc.message,
            "details": exc.details,
        }
    return {
        "error_code": "E_CONTRACT_VIOLATION",
        "message": str(exc),
        "details": {"exception_type": exc.__class__.__name__},
    }
