from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.dependencies import require_scopes
from app.schemas.common import BaseSchema
from app.tenants.context import TenantContext, get_tenant_context


class TenantMeResponse(BaseSchema):
    id: str
    name: str
    status: str


router = APIRouter(tags=["tenants"])


@router.get(
    "/tenants/me",
    response_model=TenantMeResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def get_tenant_me(
    tenant: TenantContext = Depends(get_tenant_context),
) -> TenantMeResponse:
    return TenantMeResponse(id=str(tenant.id), name=tenant.name, status=tenant.status)
