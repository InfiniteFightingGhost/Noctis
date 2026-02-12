from __future__ import annotations

from datetime import datetime

from app.schemas.common import BaseSchema


class ExperimentResponse(BaseSchema):
    id: str
    name: str
    description: str | None = None
    created_at: datetime


class ExperimentsResponse(BaseSchema):
    experiments: list[ExperimentResponse]
