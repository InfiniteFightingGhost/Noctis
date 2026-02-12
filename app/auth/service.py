from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable
import uuid

import jwt
from jwt import InvalidTokenError

from app.auth.context import AuthContext
from app.core.settings import get_settings
from app.db.models import ServiceClient, ServiceClientKey
from app.db.session import run_with_db_retry


ROLE_SCOPES: dict[str, set[str]] = {
    "ingest": {"ingest", "read"},
    "read": {"read"},
    "admin": {"ingest", "read", "admin"},
}

ALLOWED_KEY_STATUSES = {"active", "rotating"}


@dataclass(frozen=True)
class AuthError(Exception):
    code: str
    message: str
    status_code: int = 401


def authenticate_token(token: str) -> AuthContext:
    settings = get_settings()
    try:
        header = jwt.get_unverified_header(token)
    except InvalidTokenError as exc:
        raise AuthError(code="invalid_header", message="Invalid token header") from exc

    kid = header.get("kid")
    alg = header.get("alg")
    if not kid:
        raise AuthError(code="missing_kid", message="Token missing key id")
    if not alg or alg not in settings.jwt_allowed_algorithms:
        raise AuthError(code="invalid_alg", message="Unsupported token algorithm")

    def _lookup(session):
        return (
            session.query(ServiceClientKey, ServiceClient)
            .join(ServiceClient, ServiceClientKey.client_id == ServiceClient.id)
            .filter(ServiceClientKey.key_id == kid)
            .filter(ServiceClientKey.status.in_(ALLOWED_KEY_STATUSES))
            .filter(ServiceClient.status == "active")
            .first()
        )

    row = run_with_db_retry(_lookup, operation_name="auth_lookup_key")
    if not row:
        raise AuthError(code="invalid_key", message="Unknown or inactive key")
    key, client = row
    signing_key = key.public_key or key.secret
    if not signing_key:
        raise AuthError(code="missing_key", message="Key material missing")
    if alg.startswith("RS") and not key.public_key:
        raise AuthError(code="invalid_key", message="Public key required for RS tokens")
    if alg.startswith("HS") and not key.secret:
        raise AuthError(code="invalid_key", message="Secret required for HS tokens")

    try:
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=[alg],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            leeway=settings.jwt_leeway_seconds,
        )
    except InvalidTokenError as exc:
        raise AuthError(
            code="invalid_token", message="Token validation failed"
        ) from exc

    sub = payload.get("sub")
    tenant_id = payload.get("tenant_id")
    if not sub or not tenant_id:
        raise AuthError(code="missing_claim", message="Token missing required claims")
    if str(client.id) != str(sub):
        raise AuthError(code="client_mismatch", message="Token subject mismatch")
    tenant_uuid = _parse_uuid(tenant_id, "tenant_id")
    if str(client.tenant_id) != str(tenant_uuid):
        raise AuthError(code="tenant_mismatch", message="Token tenant mismatch")

    role = client.role
    scopes = ROLE_SCOPES.get(role, set())
    if not scopes:
        logging.getLogger("app.auth").warning(
            "unknown_role", extra={"role": role, "client_id": str(client.id)}
        )
    return AuthContext(
        client_id=uuid.UUID(str(client.id)),
        client_name=client.name,
        role=role,
        tenant_id=tenant_uuid,
        scopes=scopes,
        key_id=str(kid),
    )


def scopes_allow(scopes: Iterable[str], required: Iterable[str]) -> bool:
    scope_set = set(scopes)
    return set(required).issubset(scope_set)


def _parse_uuid(value: object, field_name: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except ValueError as exc:
        raise AuthError(code="invalid_claim", message=f"Invalid {field_name}") from exc
