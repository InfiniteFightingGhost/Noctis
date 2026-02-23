from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import Field

from app.schemas.common import BaseSchema


class DeviceCreate(BaseSchema):
    name: str = Field(min_length=1, max_length=200)
    external_id: str | None = Field(default=None, max_length=200)


class DeviceResponse(BaseSchema):
    id: uuid.UUID
    name: str
    external_id: str | None
    user_id: uuid.UUID | None
    created_at: datetime


class DeviceUpdate(BaseSchema):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    external_id: str | None = Field(default=None, max_length=200)


class DeviceUserLink(BaseSchema):
    user_id: uuid.UUID
