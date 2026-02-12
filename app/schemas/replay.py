from __future__ import annotations

from datetime import datetime

from app.schemas.common import BaseSchema


class ReplayRequest(BaseSchema):
    recording_id: str
    model_version_a: str
    model_version_b: str


class ReplayResponse(BaseSchema):
    recording_id: str
    model_version_a: str
    model_version_b: str
    generated_at: datetime
    summary: dict
