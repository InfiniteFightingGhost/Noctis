from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.common import BaseSchema


class DeviceEpochIngest(BaseModel):
    epoch_index: int = Field(ge=0)
    epoch_start_ts: datetime
    metrics: Any
    feature_schema_version: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: Any) -> Any:
        if isinstance(value, (list, dict, str)):
            return value
        raise ValueError("metrics must be list, dict, or base64 string")


class DeviceEpochIngestBatch(BaseSchema):
    device_id: uuid.UUID | None = None
    device_external_id: str | None = Field(default=None, max_length=200)
    device_name: str | None = Field(default=None, max_length=200)
    recording_id: uuid.UUID | None = None
    recording_started_at: datetime | None = None
    timezone: str | None = Field(default=None, max_length=64)
    forward_to_ml: bool = True
    epochs: list[DeviceEpochIngest] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_device_selector(self) -> "DeviceEpochIngestBatch":
        if self.device_id is None and not self.device_external_id:
            raise ValueError("device_id or device_external_id is required")
        return self
