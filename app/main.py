from __future__ import annotations

import logging
import time
from http import HTTPStatus
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.devices import router as devices_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.api.models import router as models_router
from app.api.account import router as account_router
from app.api.alarm import router as alarm_router
from app.api.challenges import router as challenges_router
from app.api.coach import router as coach_router
from app.api.routines import router as routines_router
from app.api.search import router as search_router
from app.api.sleep_ui import router as sleep_ui_router
from app.api.tenants import router as tenants_router
from app.api.users import router as users_router
from app.experiments.router import router as experiments_router
from app.feature_store.service import (
    ensure_active_schema_from_path,
    get_active_feature_schema,
)
from app.feature_store.router import router as feature_schemas_router
from app.promotion.router import router as promotion_router
from app.replay.router import router as replay_router
from app.api.predict import router as predict_router
from app.api.recordings import router as recordings_router
from app.auth.middleware import JWTAuthMiddleware
from app.core.logging import configure_logging
from app.core.metrics import (
    ERROR_COUNT,
    MODEL_UNAVAILABLE_COUNT,
    RATE_LIMITED_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from app.core.rate_limit import RateLimitRule, SlidingWindowRateLimiter
from app.core.settings import get_settings
from app.ml.registry import ModelRegistry
from app.db.session import run_with_db_retry
from app.drift.router import router as drift_router
from app.evaluation.router import router as evaluation_router
from app.monitoring.router import router as monitoring_router
from app.performance.router import router as performance_router
from app.stress.router import router as stress_router
from app.resilience.router import router as resilience_router
from app.timescale_ops.router import router as timescale_router
from app.audit.router import router as audit_router
from app.resilience.faults import is_fault_active
from app.user_auth.router import router as user_auth_router
from app.utils.error_payloads import error_payload
from app.utils.request_id import get_request_id, set_request_id
from app.utils.errors import AppError, ModelUnavailableError, RequestTimeoutError


@asynccontextmanager
async def lifespan(app: FastAPI):
    log = logging.getLogger("app.startup")
    settings = get_settings()

    # Wait for DB with generous retries — on Railway the managed DB can take
    # longer to accept connections than the default 3-attempt policy allows.
    db_ready = False
    for attempt in range(1, 31):
        try:

            def _validate_db(session):
                session.execute(text("SELECT 1"))

            run_with_db_retry(_validate_db, operation_name="startup_validation")
            db_ready = True
            break
        except Exception as exc:
            log.warning("startup_db_not_ready attempt=%d error=%s", attempt, exc)
            await anyio.sleep(min(2.0 * attempt, 10.0))

    if not db_ready:
        log.error("startup_db_unavailable: continuing without schema bootstrap")
    else:
        try:

            def _bootstrap_schema(session):
                ensure_active_schema_from_path(
                    session,
                    schema_path=settings.model_registry_path
                    / settings.active_model_version
                    / "feature_schema.json",
                    activate=True,
                )

            run_with_db_retry(
                _bootstrap_schema,
                commit=True,
                operation_name="feature_schema_bootstrap",
            )
        except Exception as exc:
            log.error("startup_schema_bootstrap_failed error=%s", exc)

    # Model load failure is non-fatal — get_loaded() will retry on first request.
    try:
        app.state.model_registry.load_active()
    except Exception as exc:
        log.error("startup_model_load_failed error=%s", exc)

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
    app.state.started_at = datetime.now(timezone.utc)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    app.add_middleware(
        JWTAuthMiddleware,
        exempt_paths={
            "/healthz",
            "/readyz",
            "/metrics",
            "/docs",
            "/openapi.json",
            f"{settings.api_v1_prefix}/auth/register",
            f"{settings.api_v1_prefix}/auth/register/",
            f"{settings.api_v1_prefix}/auth/login",
            f"{settings.api_v1_prefix}/auth/login/",
            f"{settings.api_v1_prefix}/auth/me",
            f"{settings.api_v1_prefix}/auth/me/",
        },
    )

    def _schema_provider():
        return run_with_db_retry(
            lambda session: get_active_feature_schema(session),
            operation_name="feature_schema_active",
        )

    app.state.model_registry = ModelRegistry(
        root=settings.model_registry_path,
        active_version=settings.active_model_version,
        schema_provider=_schema_provider,
    )
    app.state.rate_limiter = (
        SlidingWindowRateLimiter(
            window_seconds=settings.rate_limit_window_seconds,
            default_limit=settings.rate_limit_default_requests,
            rules=[
                RateLimitRule(
                    path=f"{settings.api_v1_prefix}/auth/login",
                    max_requests=settings.rate_limit_auth_requests,
                ),
                RateLimitRule(
                    path=f"{settings.api_v1_prefix}/auth/register",
                    max_requests=settings.rate_limit_auth_requests,
                ),
                RateLimitRule(
                    path=f"{settings.api_v1_prefix}/predict",
                    max_requests=settings.rate_limit_predict_requests,
                ),
                RateLimitRule(
                    path=f"{settings.api_v1_prefix}/epochs:ingest",
                    max_requests=settings.rate_limit_ingest_requests,
                ),
                RateLimitRule(
                    path=f"{settings.api_v1_prefix}/epochs:ingest-device",
                    max_requests=settings.rate_limit_ingest_requests,
                ),
            ],
        )
        if settings.rate_limit_enabled
        else None
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = set_request_id(request.headers.get("X-Request-Id"))
        start = time.perf_counter()
        status_code = HTTPStatus.INTERNAL_SERVER_ERROR
        metric_path = _metric_path_template(request)
        try:
            if is_fault_active("timeout"):
                raise RequestTimeoutError("Request timeout (fault injected)")
            if _is_rate_limited(request):
                status_code = HTTPStatus.TOO_MANY_REQUESTS
                RATE_LIMITED_COUNT.labels(path=metric_path).inc()
                response = _error_response(
                    int(status_code),
                    error_payload(
                        code="rate_limited",
                        message="Too many requests",
                        classification="client",
                    ),
                )
                response.headers["Retry-After"] = str(settings.rate_limit_window_seconds)
                response.headers["X-Request-Id"] = request_id
                response.headers["X-Correlation-Id"] = request_id
                return response
            with anyio.fail_after(settings.request_timeout_seconds):
                response = await call_next(request)
            status_code = response.status_code
        except TimeoutError as exc:
            status_code = HTTPStatus.GATEWAY_TIMEOUT
            raise RequestTimeoutError() from exc
        except Exception:
            status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            raise
        finally:
            duration = time.perf_counter() - start
            REQUEST_LATENCY.labels(path=metric_path).observe(duration)
            REQUEST_COUNT.labels(
                method=request.method,
                path=metric_path,
                status=str(int(status_code)),
            ).inc()
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Correlation-Id"] = request_id
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logging.getLogger("app").warning("validation_error", extra={"detail": str(exc)})
        payload = error_payload(
            code="validation_error",
            message=str(exc),
            classification="client",
        )
        ERROR_COUNT.labels(code="validation_error", classification="client").inc()
        return _error_response(400, payload)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        code, classification = _map_http_error(exc.status_code)
        payload = error_payload(
            code=code,
            message=str(exc.detail),
            classification=classification,
        )
        ERROR_COUNT.labels(code=code, classification=classification).inc()
        return _error_response(exc.status_code, payload)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):
        detail = jsonable_encoder(exc.errors())
        payload = error_payload(
            code="validation_error",
            message="Request validation failed",
            classification="client",
            extra={"detail": detail},
        )
        ERROR_COUNT.labels(code="validation_error", classification="client").inc()
        return _error_response(422, payload)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        if isinstance(exc, ModelUnavailableError):
            MODEL_UNAVAILABLE_COUNT.inc()
        ERROR_COUNT.labels(code=exc.detail.code, classification=exc.detail.classification).inc()
        payload = error_payload(
            code=exc.detail.code,
            message=exc.detail.message,
            classification=exc.detail.classification,
            extra=exc.detail.extra,
        )
        return _error_response(exc.detail.status_code, payload)

    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError):
        logging.getLogger("app").warning("db_error", extra={"detail": str(exc)})
        ERROR_COUNT.labels(code="db_error", classification="dependency").inc()
        payload = error_payload(
            code="db_error",
            message="Database error",
            classification="dependency",
        )
        return _error_response(503, payload)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logging.getLogger("app").exception("unhandled_error")
        ERROR_COUNT.labels(code="internal_error", classification="server").inc()
        payload = error_payload(
            code="internal_error",
            message="Unexpected error",
            classification="server",
        )
        return _error_response(500, payload)

    app.include_router(health_router)
    app.include_router(user_auth_router, prefix=settings.api_v1_prefix)
    app.include_router(devices_router, prefix=settings.api_v1_prefix)
    app.include_router(users_router, prefix=settings.api_v1_prefix)
    app.include_router(recordings_router, prefix=settings.api_v1_prefix)
    app.include_router(account_router, prefix=settings.api_v1_prefix)
    app.include_router(alarm_router, prefix=settings.api_v1_prefix)
    app.include_router(challenges_router, prefix=settings.api_v1_prefix)
    app.include_router(coach_router, prefix=settings.api_v1_prefix)
    app.include_router(routines_router, prefix=settings.api_v1_prefix)
    app.include_router(search_router, prefix=settings.api_v1_prefix)
    app.include_router(sleep_ui_router, prefix=settings.api_v1_prefix)
    app.include_router(tenants_router, prefix=settings.api_v1_prefix)
    app.include_router(ingest_router, prefix=settings.api_v1_prefix)
    app.include_router(predict_router, prefix=settings.api_v1_prefix)
    app.include_router(feature_schemas_router, prefix=settings.api_v1_prefix)
    app.include_router(models_router, prefix=settings.api_v1_prefix)
    app.include_router(experiments_router, prefix=settings.api_v1_prefix)
    app.include_router(promotion_router, prefix=settings.api_v1_prefix)
    app.include_router(replay_router, prefix=settings.api_v1_prefix)
    app.include_router(evaluation_router, prefix=settings.api_v1_prefix)
    app.include_router(drift_router, prefix=settings.api_v1_prefix)
    app.include_router(stress_router, prefix="/internal")
    app.include_router(performance_router, prefix="/internal")
    app.include_router(monitoring_router, prefix="/internal")
    app.include_router(resilience_router, prefix="/internal")
    app.include_router(timescale_router, prefix="/internal")
    app.include_router(audit_router, prefix="/internal")

    return app


def _map_http_error(status_code: int) -> tuple[str, str]:
    if status_code == 401:
        return "unauthorized", "client"
    if status_code == 403:
        return "forbidden", "client"
    if status_code == 404:
        return "not_found", "client"
    if status_code == 409:
        return "conflict", "client"
    if status_code == 422:
        return "validation_error", "client"
    if 400 <= status_code < 500:
        return "bad_request", "client"
    return "http_error", "server"


def _error_response(status_code: int, payload: dict) -> JSONResponse:
    response = JSONResponse(status_code=status_code, content=payload)
    request_id = get_request_id()
    if request_id:
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Correlation-Id"] = request_id
    return response


def _metric_path_template(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return request.url.path


def _is_rate_limited(request: Request) -> bool:
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return False
    return not limiter.allow(client_id=_client_identifier(request), path=request.url.path)


def _client_identifier(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


app = create_app()
