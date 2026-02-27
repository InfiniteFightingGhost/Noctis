from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.auth.dependencies import require_scopes
from app.db.session import run_with_db_retry
from app.schemas.coach import CoachTipResponse
from app.schemas.sleep_ui import CoachSummaryResponse
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


@router.get(
    "/coach/summary",
    response_model=CoachSummaryResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def coach_summary(
    tenant: TenantContext = Depends(get_tenant_context),
) -> CoachSummaryResponse:
    _ = tenant
    return CoachSummaryResponse(
        generated_at=datetime.now(timezone.utc),
        is_partial=False,
        insights=[
            {
                "id": "consistency",
                "title": "Keep your bedtime consistent",
                "message": "Going to bed within a 30-minute window improves recovery.",
                "tags": ["routine", "consistency"],
            }
        ],
    )
