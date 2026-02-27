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


class AlarmSoundOption(BaseSchema):
    id: str
    label: str
    mood: str | None = None


class AlarmSettingsResponse(BaseSchema):
    id: uuid.UUID
    wake_time: str
    wake_window_minutes: int
    sunrise_enabled: bool
    sunrise_intensity: int
    sound_id: str
    sound_options: list[AlarmSoundOption]
    updated_at: datetime


class AlarmSettingsUpdate(BaseSchema):
    wake_time: str | None = None
    wake_window_minutes: int | None = None
    sunrise_enabled: bool | None = None
    sunrise_intensity: int | None = None
    sound_id: str | None = None
