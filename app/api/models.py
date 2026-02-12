from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.auth.dependencies import require_admin
from app.core.metrics import MODEL_RELOAD_COUNT
from app.governance.service import record_audit_log_with_retry
from app.schemas.models import ModelReloadResponse
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["models"], dependencies=[Depends(require_admin)])


@router.post("/models/reload", response_model=ModelReloadResponse)
def reload_model(
    request: Request,
    tenant: TenantContext = Depends(get_tenant_context),
) -> ModelReloadResponse:
    registry = request.app.state.model_registry
    loaded = registry.reload()
    MODEL_RELOAD_COUNT.inc()
    record_audit_log_with_retry(
        tenant_id=tenant.id,
        actor="service_client",
        action="model_reload",
        target_type="model",
        target_id=loaded.version,
    )
    return ModelReloadResponse(model_version=loaded.version, reloaded=True)
