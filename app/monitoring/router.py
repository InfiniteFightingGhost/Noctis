from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.auth.dependencies import require_admin
from app.db.session import run_with_db_retry
from app.monitoring.service import build_monitoring_summary
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(
    tags=["monitoring"],
    dependencies=[Depends(require_admin)],
)


@router.get("/monitoring/summary")
def monitoring_summary(
    request: Request,
    tenant: TenantContext = Depends(get_tenant_context),
) -> dict[str, object]:
    def _op(session):
        return build_monitoring_summary(session, request.app.state.model_registry)

    return run_with_db_retry(_op, operation_name="monitoring_summary")
