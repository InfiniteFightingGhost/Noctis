from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.auth.dependencies import require_admin
from app.core.metrics import INFERENCE_P95_MS, INFERENCE_P99_MS, MEMORY_RSS_MB
from app.core.settings import get_settings
from app.db.session import run_with_db_retry
from app.schemas.performance import PerformanceResponse
from app.performance.service import build_performance_snapshot
from app.utils.errors import ModelUnavailableError
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["performance"], dependencies=[Depends(require_admin)])


@router.get("/performance", response_model=PerformanceResponse)
def performance_snapshot(
    request: Request,
    tenant: TenantContext = Depends(get_tenant_context),
) -> PerformanceResponse:
    settings = get_settings()

    def _op(session):
        return build_performance_snapshot(
            session,
            tenant_id=tenant.id,
            started_at=request.app.state.started_at,
            sample_size=settings.performance_sample_size,
        )

    payload = run_with_db_retry(_op, operation_name="performance_snapshot")
    try:
        model_version = request.app.state.model_registry.get_loaded().version
    except ModelUnavailableError:
        model_version = None
    MEMORY_RSS_MB.set(payload["memory_rss_mb"])
    INFERENCE_P95_MS.set(payload["inference_timing"].get("p95_latency_ms", 0.0))
    INFERENCE_P99_MS.set(payload["inference_timing"].get("p99_latency_ms", 0.0))
    return PerformanceResponse(
        timestamp=payload["timestamp"],
        model_version=model_version,
        memory_rss_mb=payload["memory_rss_mb"],
        db_pool=payload["db_pool"],
        uptime_seconds=payload["uptime_seconds"],
        circuit_breaker_state=payload["circuit_breaker_state"],
        inference_timing=payload["inference_timing"],
        db_write_speed=payload["db_write_speed"],
    )


@router.get("/performance/stats")
def performance_stats(
    request: Request,
    tenant: TenantContext = Depends(get_tenant_context),
) -> dict:
    settings = get_settings()

    def _op(session):
        return build_performance_snapshot(
            session,
            tenant_id=tenant.id,
            started_at=request.app.state.started_at,
            sample_size=settings.performance_sample_size,
        )

    return run_with_db_retry(_op, operation_name="performance_stats")
