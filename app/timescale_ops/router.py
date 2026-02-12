from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.dependencies import require_admin
from app.db.session import run_with_db_retry
from app.governance.service import record_audit_log_with_retry
from app.tenants.context import TenantContext, get_tenant_context
from app.timescale_ops.service import (
    apply_policy_actions,
    build_policy_actions,
    fetch_policy_state,
)


router = APIRouter(tags=["timescale"], dependencies=[Depends(require_admin)])


@router.get("/timescale/policies")
def timescale_policies(
    tenant: TenantContext = Depends(get_tenant_context),
) -> dict:
    payload = run_with_db_retry(fetch_policy_state, operation_name="timescale_policies")
    return payload


@router.post("/timescale/dry-run")
def timescale_dry_run(
    tenant: TenantContext = Depends(get_tenant_context),
) -> dict:
    actions = [action.__dict__ for action in build_policy_actions()]
    return {"actions": actions}


@router.post("/timescale/apply")
def timescale_apply(
    tenant: TenantContext = Depends(get_tenant_context),
) -> dict:
    applied = run_with_db_retry(
        apply_policy_actions, commit=True, operation_name="timescale_apply"
    )
    record_audit_log_with_retry(
        tenant_id=tenant.id,
        actor="service_client",
        action="timescale_policy_apply",
        target_type="timescale",
        metadata={"actions": applied},
    )
    return {"applied": applied}
