from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.devices import router as devices_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.api.models import router as models_router
from app.experiments.router import router as experiments_router
from app.promotion.router import router as promotion_router
from app.replay.router import router as replay_router
from app.api.predict import router as predict_router
from app.api.recordings import router as recordings_router
from app.auth.middleware import JWTAuthMiddleware
from app.core.logging import configure_logging
from app.core.metrics import (
    ERROR_COUNT,
    MODEL_UNAVAILABLE_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
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
from app.utils.request_id import get_request_id, set_request_id
from app.utils.errors import AppError, ModelUnavailableError, RequestTimeoutError


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_registry.load_active()

    def _validate_db(session):
        session.execute(text("SELECT 1"))

    run_with_db_retry(_validate_db, operation_name="startup_validation")
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
        exempt_paths={"/healthz", "/readyz", "/metrics", "/docs", "/openapi.json"},
    )

    app.state.model_registry = ModelRegistry(
        root=settings.model_registry_path,
        active_version=settings.active_model_version,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = set_request_id(request.headers.get("X-Request-Id"))
        start = time.perf_counter()
        try:
            if is_fault_active("timeout"):
                raise RequestTimeoutError("Request timeout (fault injected)")
            with anyio.fail_after(settings.request_timeout_seconds):
                response = await call_next(request)
        except TimeoutError as exc:
            raise RequestTimeoutError() from exc
        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(path=request.url.path).observe(duration)
        REQUEST_COUNT.labels(
            method=request.method, path=request.url.path, status=response.status_code
        ).inc()
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Correlation-Id"] = request_id
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logging.getLogger("app").warning("validation_error", extra={"detail": str(exc)})
        payload = _error_payload(
            code="validation_error",
            message=str(exc),
            classification="client",
        )
        ERROR_COUNT.labels(code="validation_error", classification="client").inc()
        return _error_response(400, payload)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        code, classification = _map_http_error(exc.status_code)
        payload = _error_payload(
            code=code,
            message=str(exc.detail),
            classification=classification,
        )
        ERROR_COUNT.labels(code=code, classification=classification).inc()
        return _error_response(exc.status_code, payload)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):
        payload = _error_payload(
            code="validation_error",
            message="Request validation failed",
            classification="client",
            extra={"detail": exc.errors()},
        )
        ERROR_COUNT.labels(code="validation_error", classification="client").inc()
        return _error_response(422, payload)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        if isinstance(exc, ModelUnavailableError):
            MODEL_UNAVAILABLE_COUNT.inc()
        ERROR_COUNT.labels(
            code=exc.detail.code, classification=exc.detail.classification
        ).inc()
        payload = _error_payload(
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
        payload = _error_payload(
            code="db_error",
            message="Database error",
            classification="dependency",
        )
        return _error_response(503, payload)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logging.getLogger("app").exception("unhandled_error")
        ERROR_COUNT.labels(code="internal_error", classification="server").inc()
        payload = _error_payload(
            code="internal_error",
            message="Unexpected error",
            classification="server",
        )
        return _error_response(500, payload)

    app.include_router(health_router)
    app.include_router(devices_router, prefix=settings.api_v1_prefix)
    app.include_router(recordings_router, prefix=settings.api_v1_prefix)
    app.include_router(ingest_router, prefix=settings.api_v1_prefix)
    app.include_router(predict_router, prefix=settings.api_v1_prefix)
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


def _error_payload(
    *,
    code: str,
    message: str,
    classification: str,
    extra: dict | None = None,
) -> dict:
    error_payload: dict[str, object] = {
        "code": code,
        "message": message,
        "classification": classification,
        "failure_classification": _failure_classification(code, classification),
        "request_id": get_request_id(),
    }
    payload: dict[str, object] = {"error": error_payload}
    if extra:
        error_payload["extra"] = extra
    return payload


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


def _failure_classification(code: str, classification: str) -> str:
    if classification in {"dependency", "transient"}:
        return "TRANSIENT"
    if code in {"request_timeout", "db_error", "db_circuit_open", "model_unavailable"}:
        return "TRANSIENT"
    return "FATAL"


def _error_response(status_code: int, payload: dict) -> JSONResponse:
    response = JSONResponse(status_code=status_code, content=payload)
    request_id = get_request_id()
    if request_id:
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Correlation-Id"] = request_id
    return response


app = create_app()
