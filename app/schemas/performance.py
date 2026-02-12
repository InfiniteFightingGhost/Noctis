from __future__ import annotations

from datetime import datetime

from app.schemas.common import BaseSchema


class PerformanceResponse(BaseSchema):
    timestamp: datetime
    model_version: str | None
    memory_rss_mb: float
    db_pool: dict[str, int | str]
    uptime_seconds: float
    circuit_breaker_state: str
    inference_timing: dict[str, float | int]
    db_write_speed: dict[str, float | int]
