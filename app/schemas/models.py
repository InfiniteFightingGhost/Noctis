from __future__ import annotations

from datetime import datetime

from app.schemas.common import BaseSchema


class ModelReloadResponse(BaseSchema):
    model_version: str
    reloaded: bool


class ModelVersionResponse(BaseSchema):
    version: str
    status: str
    metrics: dict | None = None
    feature_schema_version: str | None = None
    artifact_path: str | None = None
    created_at: datetime | None = None
    promoted_at: datetime | None = None
    promoted_by: str | None = None
    details: dict | None = None


class ModelVersionsResponse(BaseSchema):
    models: list[ModelVersionResponse]
