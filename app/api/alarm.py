from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.dependencies import require_scopes
from app.db.session import run_with_db_retry
from app.schemas.alarm import AlarmResponse
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["alarm"])


@router.get(
    "/alarm",
    response_model=list[AlarmResponse],
    dependencies=[Depends(require_scopes("read"))],
)
def list_alarms(
    tenant: TenantContext = Depends(get_tenant_context),
) -> list[AlarmResponse]:
    def _op(_session):
        _ = tenant
        return []

    return run_with_db_retry(_op, operation_name="alarm_list")
