from __future__ import annotations

import logging
import uuid

import anyio
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from app.auth.api_key import authenticate_hardware_api_key
from app.auth.service import AuthError, authenticate_token
from app.auth.context import AuthContext
from app.core.metrics import AUTH_FAILURE_COUNT
from app.core.settings import get_settings
from app.user_auth.security import verify_access_token
from app.utils.error_payloads import error_payload
from app.utils.request_id import get_request_id, set_request_id


class JWTAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, exempt_paths: set[str] | None = None) -> None:
        super().__init__(app)
        self._exempt_paths = exempt_paths or set()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == "OPTIONS":
            return await call_next(request)
        normalized_path = request.url.path.rstrip("/") or "/"
        if normalized_path in self._exempt_paths or request.url.path in self._exempt_paths:
            return await call_next(request)

        # 1. Try Hardware API Key first for specific paths (ingest/start)
        # This ensures that hardware devices sending X-API-Key are not blocked by
        # garbage Authorization headers (e.g. from proxies).
        settings = get_settings()
        if request.headers.get(settings.api_key_header):
            try:
                api_key_auth = authenticate_hardware_api_key(request)
                if api_key_auth is not None:
                    request.state.auth = api_key_auth
                    return await call_next(request)
            except AuthError as exc:
                # If they provided an API key and it's INVALID, fail immediately
                _log_auth_failure(exc.code, request)
                response = JSONResponse(
                    status_code=exc.status_code,
                    content=error_payload(
                        code=exc.code,
                        message=exc.message,
                        classification="client",
                        ensure_request_id=True,
                    ),
                )
                _set_request_headers(response)
                return response

        # 2. Try JWT Authorization
        authorization = request.headers.get(settings.auth_header)
        if not authorization:
            # If no Authorization header and we didn't already succeed with API Key
            _log_auth_failure("missing_token", request)
            response = JSONResponse(
                status_code=401,
                content=error_payload(
                    code="missing_token",
                    message="Missing Authorization header",
                    classification="client",
                    ensure_request_id=True,
                ),
            )
            _set_request_headers(response)
            return response

        if not authorization.startswith("Bearer "):
            _log_auth_failure("invalid_scheme", request)
            response = JSONResponse(
                status_code=401,
                content=error_payload(
                    code="invalid_scheme",
                    message="Invalid Authorization scheme",
                    classification="client",
                    ensure_request_id=True,
                ),
            )
            _set_request_headers(response)
            return response
        token = authorization.replace("Bearer ", "", 1).strip()
        try:
            auth = await anyio.to_thread.run_sync(authenticate_token, token)
        except AuthError as exc:
            user_auth = _try_authenticate_user_token(token)
            if user_auth is None:
                _log_auth_failure(exc.code, request)
                response = JSONResponse(
                    status_code=exc.status_code,
                    content=error_payload(
                        code=exc.code,
                        message=exc.message,
                        classification="client",
                        ensure_request_id=True,
                    ),
                )
                _set_request_headers(response)
                return response
            auth = user_auth
        request.state.auth = auth
        return await call_next(request)


def _log_auth_failure(reason: str, request: Request) -> None:
    request_id = get_request_id() or set_request_id(None)
    logging.getLogger("app.auth").warning(
        "auth_failure",
        extra={"reason": reason, "path": request.url.path, "request_id": request_id},
    )
    AUTH_FAILURE_COUNT.labels(reason=reason).inc()


def _set_request_headers(response: JSONResponse) -> None:
    request_id = get_request_id() or set_request_id(None)
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Correlation-Id"] = request_id


def _try_authenticate_user_token(token: str) -> AuthContext | None:
    try:
        claims = verify_access_token(token)
    except Exception:  # noqa: BLE001
        return None
    settings = get_settings()
    return AuthContext(
        client_id=claims.subject,
        client_name=claims.email,
        role="user",
        tenant_id=uuid.UUID(settings.default_tenant_id),
        scopes={"read", "ingest"},
        key_id="user-auth",
        principal_type="user",
    )
