from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_db


router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request, db: Session = Depends(get_db)) -> dict:
    db.execute(text("SELECT 1"))
    request.app.state.model_registry.get_loaded()
    return {"status": "ready"}


@router.get("/metrics")
def metrics() -> Response:
    from prometheus_client import generate_latest

    return Response(generate_latest(), media_type="text/plain")
