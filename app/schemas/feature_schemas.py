from __future__ import annotations

import uuid
from datetime import datetime

from app.schemas.common import BaseSchema


class FeatureSchemaFeatureResponse(BaseSchema):
    name: str
    dtype: str
    allowed_range: dict[str, float] | None = None
    description: str | None = None
    introduced_in_version: str
    deprecated_in_version: str | None = None
    position: int


class FeatureSchemaResponse(BaseSchema):
    id: uuid.UUID
    version: str
    hash: str
    description: str | None = None
    is_active: bool
    created_at: datetime
    features: list[FeatureSchemaFeatureResponse]


class FeatureSchemasResponse(BaseSchema):
    schemas: list[FeatureSchemaResponse]
