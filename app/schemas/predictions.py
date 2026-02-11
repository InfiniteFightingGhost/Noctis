from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import BaseSchema
from app.schemas.epochs import EpochIngest


class PredictRequest(BaseSchema):
    recording_id: uuid.UUID
    epochs: list[EpochIngest] | None = None


class PredictionItem(BaseSchema):
    window_start_ts: datetime
    window_end_ts: datetime
    predicted_stage: str
    confidence: float
    probabilities: dict[str, float]


class PredictResponse(BaseSchema):
    recording_id: uuid.UUID
    model_version: str
    feature_schema_version: str
    predictions: list[PredictionItem]


class PredictionResponse(BaseSchema):
    id: uuid.UUID
    recording_id: uuid.UUID
    window_start_ts: datetime
    window_end_ts: datetime
    model_version: str
    feature_schema_version: str
    predicted_stage: str
    confidence: float
    probabilities: dict[str, float]
    created_at: datetime
