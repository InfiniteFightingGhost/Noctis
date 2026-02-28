from __future__ import annotations

import os

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.db.session import run_with_db_retry
from app.utils.errors import ModelUnavailableError


router = APIRouter(tags=["health"])


import logging
import time

logger = logging.getLogger(__name__)


@router.get("/healthz")
def healthz() -> dict:
    start_time = time.time()
    try:
        logger.info("Health check called")
        # Quick test to ensure the app is responsive
        return {"status": "ok", "response_time": time.time() - start_time}
    except Exception:
        return {"status": "error", "response_time": time.time() - start_time}


@router.get("/readyz")
def readyz(request: Request) -> Response:
    def _op(session):
        session.execute(text("SELECT 1"))

    try:
        run_with_db_retry(_op, operation_name="readyz")
    except Exception:
        return JSONResponse(status_code=503, content={"status": "degraded", "database": "down"})
    try:
        request.app.state.model_registry.get_loaded()
    except ModelUnavailableError:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "model": "unavailable"},
        )
    return JSONResponse(content={"status": "ready"})


@router.get("/metrics")
def metrics() -> Response:
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

    multiproc_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR")
    if multiproc_dir:
        from prometheus_client import multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
