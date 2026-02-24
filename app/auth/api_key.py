from __future__ import annotations

import hmac
import uuid

from fastapi import Request

from app.auth.context import AuthContext
from app.auth.service import AuthError
from app.core.settings import get_settings


def authenticate_hardware_api_key(request: Request) -> AuthContext | None:
    settings = get_settings()
    ingest_path = f"{settings.api_v1_prefix}/epochs:ingest-device"
    if request.url.path != ingest_path:
        return None
    provided_key = request.headers.get(settings.api_key_header)
    if provided_key is None:
        return None
    if not hmac.compare_digest(provided_key, settings.api_key):
        raise AuthError(code="invalid_api_key", message="Invalid API key")
    tenant_id = uuid.UUID(settings.default_tenant_id)
    return AuthContext(
        client_id=uuid.uuid5(uuid.NAMESPACE_URL, f"hardware-api-key:{tenant_id}"),
        client_name="hardware-api-key",
        role="ingest",
        tenant_id=tenant_id,
        scopes={"ingest", "read"},
        key_id="hardware-api-key",
        principal_type="service",
    )
