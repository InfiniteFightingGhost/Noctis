from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.dependencies import require_scopes
from app.db.session import run_with_db_retry
from app.schemas.coach import CoachTipResponse
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["coach"])


@router.get(
    "/coach",
    response_model=list[CoachTipResponse],
    dependencies=[Depends(require_scopes("read"))],
)
def list_coach_tips(
    tenant: TenantContext = Depends(get_tenant_context),
) -> list[CoachTipResponse]:
    def _op(_session):
        _ = tenant
        return []

    return run_with_db_retry(_op, operation_name="coach_list")
