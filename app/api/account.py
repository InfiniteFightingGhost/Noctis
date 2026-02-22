from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.context import AuthContext
from app.auth.dependencies import get_auth_context, require_scopes
from app.schemas.account import AccountMeResponse
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["account"])


@router.get(
    "/account/me",
    response_model=AccountMeResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def get_account_me(
    auth: AuthContext = Depends(get_auth_context),
    tenant: TenantContext = Depends(get_tenant_context),
) -> AccountMeResponse:
    return AccountMeResponse(
        client_id=auth.client_id,
        client_name=auth.client_name,
        role=auth.role,
        tenant_id=tenant.id,
        tenant_name=tenant.name,
        scopes=sorted(auth.scopes),
    )
