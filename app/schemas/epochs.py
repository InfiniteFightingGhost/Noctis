from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import BaseSchema


class EpochIngest(BaseModel):
    epoch_index: int = Field(ge=0)
    epoch_start_ts: datetime
    feature_schema_version: str = Field(min_length=1, max_length=64)
    features: Any

    @field_validator("features")
    @classmethod
    def validate_features(cls, value: Any) -> Any:
        if isinstance(value, (list, dict, str)):
            return value
        raise ValueError("features must be list, dict, or base64 string")


class EpochIngestBatch(BaseSchema):
    recording_id: uuid.UUID
    epochs: list[EpochIngest] = Field(min_length=1)


class EpochResponse(BaseSchema):
    recording_id: uuid.UUID
    epoch_index: int
    epoch_start_ts: datetime
    feature_schema_version: str
    features_payload: dict
