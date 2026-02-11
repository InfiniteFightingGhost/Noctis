from __future__ import annotations

import uuid
from contextvars import ContextVar


_request_id: ContextVar[str] = ContextVar("request_id", default="")


def set_request_id(value: str | None = None) -> str:
    request_id = value or uuid.uuid4().hex
    _request_id.set(request_id)
    return request_id


def get_request_id() -> str:
    return _request_id.get()
