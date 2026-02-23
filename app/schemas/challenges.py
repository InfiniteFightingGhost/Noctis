from __future__ import annotations

import uuid
from datetime import datetime

from app.schemas.common import BaseSchema


class ChallengeResponse(BaseSchema):
    id: uuid.UUID
    name: str
    status: str
    progress: float
    starts_at: datetime | None
    ends_at: datetime | None
