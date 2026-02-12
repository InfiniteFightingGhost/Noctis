from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.auth.dependencies import require_admin
from app.db.session import run_with_db_retry
from app.governance.service import record_audit_log_with_retry
from app.promotion.service import archive_model, promote_model, rollback_model
from app.schemas.promotion import PromotionRequest, PromotionResponse
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["promotion"], dependencies=[Depends(require_admin)])


@router.post("/models/{version}/promote", response_model=PromotionResponse)
def promote(
    version: str,
    payload: PromotionRequest,
    request: Request,
    tenant: TenantContext = Depends(get_tenant_context),
) -> PromotionResponse:
    def _op(session):
        return promote_model(
            session,
            version=version,
            actor=payload.actor,
            reason=payload.reason,
        )

    model = run_with_db_retry(_op, commit=True, operation_name="promote_model")
    request.app.state.model_registry.reload()
    record_audit_log_with_retry(
        tenant_id=tenant.id,
        actor=payload.actor,
        action="model_promote",
        target_type="model",
        target_id=version,
        metadata={"reason": payload.reason},
    )
    return PromotionResponse(
        version=model.version,
        status=model.status,
        promoted_at=model.promoted_at,
        promoted_by=model.promoted_by,
    )


@router.post("/models/{version}/archive", response_model=PromotionResponse)
def archive(
    version: str,
    payload: PromotionRequest,
    tenant: TenantContext = Depends(get_tenant_context),
) -> PromotionResponse:
    def _op(session):
        return archive_model(
            session,
            version=version,
            actor=payload.actor,
            reason=payload.reason,
        )

    model = run_with_db_retry(_op, commit=True, operation_name="archive_model")
    record_audit_log_with_retry(
        tenant_id=tenant.id,
        actor=payload.actor,
        action="model_archive",
        target_type="model",
        target_id=version,
        metadata={"reason": payload.reason},
    )
    return PromotionResponse(
        version=model.version,
        status=model.status,
        promoted_at=model.promoted_at,
        promoted_by=model.promoted_by,
    )


@router.post("/models/{version}/rollback", response_model=PromotionResponse)
def rollback(
    version: str,
    payload: PromotionRequest,
    request: Request,
    tenant: TenantContext = Depends(get_tenant_context),
) -> PromotionResponse:
    def _op(session):
        return rollback_model(
            session,
            version=version,
            actor=payload.actor,
            reason=payload.reason,
        )

    model = run_with_db_retry(_op, commit=True, operation_name="rollback_model")
    request.app.state.model_registry.reload()
    record_audit_log_with_retry(
        tenant_id=tenant.id,
        actor=payload.actor,
        action="model_rollback",
        target_type="model",
        target_id=version,
        metadata={"reason": payload.reason},
    )
    return PromotionResponse(
        version=model.version,
        status=model.status,
        promoted_at=model.promoted_at,
        promoted_by=model.promoted_by,
    )
