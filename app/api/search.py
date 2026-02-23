from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.auth.dependencies import require_scopes
from app.db.session import run_with_db_retry
from app.schemas.search import SearchResponse
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["search"])


@router.get(
    "/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def search(
    q: str = Query(..., min_length=1, max_length=200),
    tenant: TenantContext = Depends(get_tenant_context),
) -> SearchResponse:
    def _op(_session):
        _ = tenant
        return SearchResponse(query=q, results=[])

    return run_with_db_retry(_op, operation_name="search")
