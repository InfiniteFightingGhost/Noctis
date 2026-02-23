from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.dependencies import require_scopes
from app.db.session import run_with_db_retry
from app.schemas.routines import RoutineResponse
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["routines"])


@router.get(
    "/routines",
    response_model=list[RoutineResponse],
    dependencies=[Depends(require_scopes("read"))],
)
def list_routines(
    tenant: TenantContext = Depends(get_tenant_context),
) -> list[RoutineResponse]:
    def _op(_session):
        _ = tenant
        return []

    return run_with_db_retry(_op, operation_name="routines_list")
