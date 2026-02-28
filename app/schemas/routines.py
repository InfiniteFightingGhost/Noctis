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


class RoutineStepResponse(BaseSchema):
    id: uuid.UUID
    title: str
    duration_minutes: int
    emoji: str | None


class RoutineCurrentResponse(BaseSchema):
    id: uuid.UUID
    title: str
    total_minutes: int
    steps: list[RoutineStepResponse]
    updated_at: datetime


class RoutineStepUpdate(BaseSchema):
    id: uuid.UUID | None = None
    title: str
    duration_minutes: int
    emoji: str | None = None


class RoutineCurrentUpdate(BaseSchema):
    title: str | None = None
    steps: list[RoutineStepUpdate] | None = None
