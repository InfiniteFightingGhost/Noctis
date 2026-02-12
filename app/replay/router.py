from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.dependencies import require_scopes
from app.db.session import run_with_db_retry
from app.replay.service import replay_models
from app.schemas.replay import ReplayRequest, ReplayResponse
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["replay"], dependencies=[Depends(require_scopes("read"))])


@router.post("/models/replay", response_model=ReplayResponse)
def replay(
    payload: ReplayRequest,
    tenant: TenantContext = Depends(get_tenant_context),
) -> ReplayResponse:
    def _op(session):
        return replay_models(
            session,
            tenant_id=tenant.id,
            recording_id=payload.recording_id,
            model_version_a=payload.model_version_a,
            model_version_b=payload.model_version_b,
        )

    result = run_with_db_retry(_op, operation_name="replay_models")
    return ReplayResponse(**result)
