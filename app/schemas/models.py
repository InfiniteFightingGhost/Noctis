from __future__ import annotations

from app.schemas.common import BaseSchema


class ModelReloadResponse(BaseSchema):
    model_version: str
    reloaded: bool
