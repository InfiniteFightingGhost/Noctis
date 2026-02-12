from __future__ import annotations

import logging
import uuid

from fastapi import Depends, HTTPException, Request, status

from app.auth.context import AuthContext
from app.auth.service import AuthError, authenticate_token, scopes_allow
from app.core.metrics import AUTH_FAILURE_COUNT
from app.core.settings import get_settings
from app.governance.service import record_audit_log_with_retry
from app.utils.request_id import get_request_id


def get_auth_context(
    request: Request,
) -> AuthContext:
    existing = getattr(request.state, "auth", None)
    if isinstance(existing, AuthContext):
        return existing
    authorization = request.headers.get(get_settings().auth_header)
    if not authorization:
        _handle_auth_failure("missing_token", request, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    if not authorization.startswith("Bearer "):
        _handle_auth_failure("invalid_scheme", request, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization scheme",
        )
    token = authorization.replace("Bearer ", "", 1).strip()
    try:
        auth = authenticate_token(token)
    except AuthError as exc:
        _handle_auth_failure(exc.code, request, None)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    request.state.auth = auth
    return auth


def require_scopes(*required: str):
    def _dependency(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
        if required and not scopes_allow(auth.scopes, required):
            _handle_auth_failure("insufficient_scope", None, auth.tenant_id)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient scope",
            )
        return auth

    return _dependency


def require_admin(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
    if auth.role != "admin":
        _handle_auth_failure("admin_required", None, auth.tenant_id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required"
        )
    return auth


def _handle_auth_failure(
    reason: str, request: Request | None, tenant_id: uuid.UUID | None
) -> None:
    request_id = get_request_id()
    path = request.url.path if request else None
    logging.getLogger("app.auth").warning(
        "auth_failure",
        extra={"reason": reason, "path": path, "request_id": request_id},
    )
    AUTH_FAILURE_COUNT.labels(reason=reason).inc()
    if tenant_id is None:
        settings = get_settings()
        try:
            tenant_id = uuid.UUID(settings.default_tenant_id)
        except ValueError:
            tenant_id = None
    if tenant_id:
        try:
            record_audit_log_with_retry(
                tenant_id=tenant_id,
                actor="system",
                action="auth_failure",
                target_type="service_client",
                metadata={"reason": reason, "path": path, "request_id": request_id},
            )
        except Exception:  # noqa: BLE001
            logging.getLogger("app.auth").warning("auth_audit_log_failed")
