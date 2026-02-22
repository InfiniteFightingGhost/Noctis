from __future__ import annotations

import uuid
from datetime import datetime

from app.schemas.common import BaseSchema


class AlarmResponse(BaseSchema):
    id: uuid.UUID
    name: str
    scheduled_for: datetime
    enabled: bool
    created_at: datetime
