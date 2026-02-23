from __future__ import annotations

import uuid
from datetime import datetime

from app.schemas.common import BaseSchema


class CoachTipResponse(BaseSchema):
    id: uuid.UUID
    title: str
    message: str
    status: str
    created_at: datetime
