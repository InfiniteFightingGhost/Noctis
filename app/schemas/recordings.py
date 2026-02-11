from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import Field

from app.schemas.common import BaseSchema


class RecordingCreate(BaseSchema):
    device_id: uuid.UUID
    started_at: datetime
    timezone: str | None = Field(default=None, max_length=64)


class RecordingResponse(BaseSchema):
    id: uuid.UUID
    device_id: uuid.UUID
    started_at: datetime
    ended_at: datetime | None
    timezone: str | None
    created_at: datetime
