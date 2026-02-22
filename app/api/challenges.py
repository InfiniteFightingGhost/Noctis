from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.dependencies import require_scopes
from app.db.session import run_with_db_retry
from app.schemas.challenges import ChallengeResponse
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["challenges"])


@router.get(
    "/challenges",
    response_model=list[ChallengeResponse],
    dependencies=[Depends(require_scopes("read"))],
)
def list_challenges(
    tenant: TenantContext = Depends(get_tenant_context),
) -> list[ChallengeResponse]:
    def _op(_session):
        _ = tenant
        return []

    return run_with_db_retry(_op, operation_name="challenges_list")
