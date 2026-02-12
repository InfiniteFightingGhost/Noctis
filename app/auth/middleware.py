from __future__ import annotations

import logging
import uuid

import anyio
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from app.auth.service import AuthError, authenticate_token
from app.core.metrics import AUTH_FAILURE_COUNT
from app.core.settings import get_settings
from app.governance.service import record_audit_log_with_retry
from app.utils.request_id import get_request_id, set_request_id


class JWTAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, exempt_paths: set[str] | None = None) -> None:
        super().__init__(app)
        self._exempt_paths = exempt_paths or set()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path in self._exempt_paths:
            return await call_next(request)
        authorization = request.headers.get(get_settings().auth_header)
        if not authorization:
            _log_auth_failure("missing_token", request)
            response = JSONResponse(
                status_code=401, content={"detail": "Missing Authorization header"}
            )
            _set_request_headers(response)
            return response
        if not authorization.startswith("Bearer "):
            _log_auth_failure("invalid_scheme", request)
            response = JSONResponse(
                status_code=401, content={"detail": "Invalid Authorization scheme"}
            )
            _set_request_headers(response)
            return response
        token = authorization.replace("Bearer ", "", 1).strip()
        try:
            auth = await anyio.to_thread.run_sync(authenticate_token, token)
        except AuthError as exc:
            _log_auth_failure(exc.code, request)
            response = JSONResponse(
                status_code=exc.status_code, content={"detail": exc.message}
            )
            _set_request_headers(response)
            return response
        request.state.auth = auth
        return await call_next(request)


def _log_auth_failure(reason: str, request: Request) -> None:
    request_id = get_request_id() or set_request_id(None)
    logging.getLogger("app.auth").warning(
        "auth_failure",
        extra={"reason": reason, "path": request.url.path, "request_id": request_id},
    )
    AUTH_FAILURE_COUNT.labels(reason=reason).inc()
    settings = get_settings()
    try:
        tenant_id = settings.default_tenant_id
    except Exception:  # noqa: BLE001
        tenant_id = None
    if tenant_id:
        try:
            record_audit_log_with_retry(
                tenant_id=uuid.UUID(tenant_id),
                actor="system",
                action="auth_failure",
                target_type="service_client",
                metadata={
                    "reason": reason,
                    "path": request.url.path,
                    "request_id": request_id,
                },
            )
        except Exception:  # noqa: BLE001
            logging.getLogger("app.auth").warning("auth_audit_log_failed")


def _set_request_headers(response: JSONResponse) -> None:
    request_id = get_request_id() or set_request_id(None)
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Correlation-Id"] = request_id
