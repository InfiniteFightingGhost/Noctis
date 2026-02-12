from __future__ import annotations

from datetime import datetime

from app.schemas.common import BaseSchema


class PromotionRequest(BaseSchema):
    actor: str
    reason: str | None = None


class PromotionResponse(BaseSchema):
    version: str
    status: str
    promoted_at: datetime | None = None
    promoted_by: str | None = None
