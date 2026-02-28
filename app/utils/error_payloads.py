from __future__ import annotations

from app.utils.request_id import get_request_id, set_request_id


def failure_classification(code: str, classification: str) -> str:
    if classification in {"dependency", "transient"}:
        return "TRANSIENT"
    if code in {"request_timeout", "db_error", "db_circuit_open", "model_unavailable"}:
        return "TRANSIENT"
    return "FATAL"


def error_payload(
    *,
    code: str,
    message: str,
    classification: str,
    extra: dict | None = None,
    ensure_request_id: bool = False,
) -> dict:
    request_id = get_request_id()
    if ensure_request_id and request_id is None:
        request_id = set_request_id(None)

    payload: dict[str, object] = {
        "code": code,
        "message": message,
        "classification": classification,
        "failure_classification": failure_classification(code, classification),
        "request_id": request_id,
    }
    if extra:
        payload["extra"] = extra
    return {"error": payload}
