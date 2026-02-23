from __future__ import annotations

import uuid

from app.schemas.common import BaseSchema


class AccountMeResponse(BaseSchema):
    client_id: uuid.UUID
    client_name: str
    role: str
    tenant_id: uuid.UUID
    tenant_name: str
    scopes: list[str]
