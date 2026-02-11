from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.api.devices import router as devices_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.api.models import router as models_router
from app.api.predict import router as predict_router
from app.api.recordings import router as recordings_router
from app.core.logging import configure_logging
from app.core.metrics import REQUEST_COUNT, REQUEST_LATENCY
from app.core.settings import get_settings
from app.ml.registry import ModelRegistry
from app.utils.request_id import set_request_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_registry.load_active()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    app.state.model_registry = ModelRegistry(
        root=settings.model_registry_path,
        active_version=settings.active_model_version,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = set_request_id(request.headers.get("X-Request-Id"))
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(path=request.url.path).observe(duration)
        REQUEST_COUNT.labels(
            method=request.method, path=request.url.path, status=response.status_code
        ).inc()
        response.headers["X-Request-Id"] = request_id
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logging.getLogger("app").warning("validation_error", extra={"detail": str(exc)})
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    app.include_router(health_router)
    app.include_router(devices_router, prefix=settings.api_v1_prefix)
    app.include_router(recordings_router, prefix=settings.api_v1_prefix)
    app.include_router(ingest_router, prefix=settings.api_v1_prefix)
    app.include_router(predict_router, prefix=settings.api_v1_prefix)
    app.include_router(models_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
