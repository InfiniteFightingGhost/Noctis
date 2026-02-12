from __future__ import annotations

from datetime import datetime
from typing import Any

from app.schemas.common import BaseSchema


class FaultEnableRequest(BaseSchema):
    name: str
    ttl_seconds: int | None = 60
    params: dict[str, Any] | None = None


class FaultState(BaseSchema):
    name: str
    enabled_at: datetime
    expires_at: datetime | None
    params: dict[str, Any]


class FaultListResponse(BaseSchema):
    faults: list[FaultState]


class FaultDisableResponse(BaseSchema):
    removed: bool
