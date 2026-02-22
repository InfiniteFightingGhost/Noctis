from __future__ import annotations

import uuid
from datetime import datetime

from app.schemas.common import BaseSchema


class RoutineResponse(BaseSchema):
    id: uuid.UUID
    name: str
    description: str | None
    status: str
    created_at: datetime
