import hmac
import re
import uuid

from fastapi import Request

from app.auth.context import AuthContext
from app.auth.service import AuthError
from app.core.settings import get_settings


def authenticate_hardware_api_key(request: Request) -> AuthContext | None:
    settings = get_settings()
    path = request.url.path
    prefix = settings.api_v1_prefix

    # Define whitelist patterns
    allowed_patterns = [
        rf"^{prefix}/epochs:ingest-device/?$",
        rf"^{prefix}/recordings:start/?$",
        rf"^{prefix}/sleep/latest/summary/?$",
        rf"^{prefix}/recordings/?$",
        rf"^{prefix}/recordings/[a-f0-9\-]+/?$",
        rf"^{prefix}/recordings/[a-f0-9\-]+/epochs/?$",
        rf"^{prefix}/recordings/[a-f0-9\-]+/predictions/?$",
    ]

    is_allowed = any(re.match(pattern, path, re.IGNORECASE) for pattern in allowed_patterns)
    if not is_allowed:
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
        role="admin",  # Escalating to admin to ensure read/ingest across recordings
        tenant_id=tenant_id,
        scopes={"ingest", "read", "admin"},
        key_id="hardware-api-key",
        principal_type="service",
    )
