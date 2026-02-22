from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import Field

from app.schemas.common import BaseSchema


class UserCreate(BaseSchema):
    name: str = Field(min_length=1, max_length=200)
    external_id: str | None = Field(default=None, max_length=200)


class UserResponse(BaseSchema):
    id: uuid.UUID
    name: str
    external_id: str | None
    created_at: datetime
