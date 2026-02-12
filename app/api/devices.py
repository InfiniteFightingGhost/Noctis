from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.dependencies import require_scopes
from app.db.models import Device
from app.db.session import run_with_db_retry
from app.schemas.devices import DeviceCreate, DeviceResponse
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["devices"], dependencies=[Depends(require_scopes("ingest"))])


@router.post("/devices", response_model=DeviceResponse)
def create_device(
    payload: DeviceCreate,
    tenant: TenantContext = Depends(get_tenant_context),
) -> DeviceResponse:
    def _op(session):
        device = Device(
            tenant_id=tenant.id, name=payload.name, external_id=payload.external_id
        )
        session.add(device)
        session.flush()
        session.refresh(device)
        return device

    device = run_with_db_retry(_op, commit=True, operation_name="create_device")
    return DeviceResponse.model_validate(device)
