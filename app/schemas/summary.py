from __future__ import annotations

from datetime import datetime

from app.schemas.common import BaseSchema


class RecordingSummary(BaseSchema):
    recording_id: str
    from_ts: datetime
    to_ts: datetime
    total_minutes: float
    time_in_stage_minutes: dict[str, float]
    sleep_latency_minutes: float | None
    waso_minutes: float | None
