from __future__ import annotations

from fastapi import APIRouter, Request, Response
from sqlalchemy import text

from app.db.session import run_with_db_retry
from app.utils.errors import ModelUnavailableError


router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request) -> dict:
    def _op(session):
        session.execute(text("SELECT 1"))

    run_with_db_retry(_op, operation_name="readyz")
    try:
        request.app.state.model_registry.get_loaded()
    except ModelUnavailableError:
        return {"status": "degraded", "model": "unavailable"}
    return {"status": "ready"}


@router.get("/metrics")
def metrics() -> Response:
    from prometheus_client import generate_latest

    return Response(generate_latest(), media_type="text/plain")
