from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import require_admin
from app.governance.service import record_audit_log_with_retry
from app.resilience.faults import disable_fault, enable_fault, list_faults
from app.schemas.resilience import (
    FaultDisableResponse,
    FaultEnableRequest,
    FaultListResponse,
    FaultState,
)
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["resilience"], dependencies=[Depends(require_admin)])


@router.post("/faults/enable", response_model=FaultListResponse)
def enable_fault_endpoint(
    payload: FaultEnableRequest,
    tenant: TenantContext = Depends(get_tenant_context),
) -> FaultListResponse:
    params = payload.params or {}
    if payload.name == "db_latency_ms":
        latency_ms = int(params.get("latency_ms") or 0)
        if latency_ms <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="latency_ms must be positive",
            )
    enable_fault(payload.name, ttl_seconds=payload.ttl_seconds, params=params)
    record_audit_log_with_retry(
        tenant_id=tenant.id,
        actor="service_client",
        action="fault_enable",
        target_type="fault",
        target_id=payload.name,
        metadata={"ttl_seconds": payload.ttl_seconds, "params": params},
    )
    faults: list[FaultState] = [
        FaultState.model_validate(fault) for fault in list_faults()
    ]
    return FaultListResponse(faults=faults)


@router.post("/faults/disable", response_model=FaultDisableResponse)
def disable_fault_endpoint(
    payload: FaultEnableRequest,
    tenant: TenantContext = Depends(get_tenant_context),
) -> FaultDisableResponse:
    removed = disable_fault(payload.name)
    record_audit_log_with_retry(
        tenant_id=tenant.id,
        actor="service_client",
        action="fault_disable",
        target_type="fault",
        target_id=payload.name,
    )
    return FaultDisableResponse(removed=removed)


@router.get("/faults", response_model=FaultListResponse)
def list_faults_endpoint(
    tenant: TenantContext = Depends(get_tenant_context),
) -> FaultListResponse:
    faults: list[FaultState] = [
        FaultState.model_validate(fault) for fault in list_faults()
    ]
    return FaultListResponse(faults=faults)
