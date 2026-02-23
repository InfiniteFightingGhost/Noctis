from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import require_admin, require_scopes
from app.db.models import Device, User
from app.db.session import run_with_db_retry
from app.schemas.devices import (
    DeviceCreate,
    DeviceResponse,
    DeviceUpdate,
    DeviceUserLink,
)
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["devices"])


@router.post("/devices", response_model=DeviceResponse)
def create_device(
    payload: DeviceCreate,
    tenant: TenantContext = Depends(get_tenant_context),
    _auth=Depends(require_scopes("ingest")),
) -> DeviceResponse:
    def _op(session):
        device = Device(tenant_id=tenant.id, name=payload.name, external_id=payload.external_id)
        session.add(device)
        session.flush()
        session.refresh(device)
        return device

    device = run_with_db_retry(_op, commit=True, operation_name="create_device")
    return DeviceResponse.model_validate(device)


@router.get(
    "/devices",
    response_model=list[DeviceResponse],
    dependencies=[Depends(require_scopes("read"))],
)
def list_devices(
    tenant: TenantContext = Depends(get_tenant_context),
) -> list[DeviceResponse]:
    def _op(session):
        return (
            session.query(Device)
            .filter(Device.tenant_id == tenant.id)
            .order_by(Device.created_at.desc())
            .all()
        )

    devices = run_with_db_retry(_op, operation_name="devices_list")
    return [DeviceResponse.model_validate(device) for device in devices]


@router.get(
    "/devices/{device_id}",
    response_model=DeviceResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def get_device(
    device_id: uuid.UUID,
    tenant: TenantContext = Depends(get_tenant_context),
) -> DeviceResponse:
    def _op(session):
        return (
            session.query(Device)
            .filter(Device.id == device_id)
            .filter(Device.tenant_id == tenant.id)
            .one_or_none()
        )

    device = run_with_db_retry(_op, operation_name="device_get")
    if not device:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    return DeviceResponse.model_validate(device)


@router.put(
    "/devices/{device_id}",
    response_model=DeviceResponse,
    dependencies=[Depends(require_scopes("ingest"))],
)
def update_device(
    device_id: uuid.UUID,
    payload: DeviceUpdate,
    tenant: TenantContext = Depends(get_tenant_context),
) -> DeviceResponse:
    def _op(session):
        device = (
            session.query(Device)
            .filter(Device.id == device_id)
            .filter(Device.tenant_id == tenant.id)
            .one_or_none()
        )
        if not device:
            return None
        if "name" in payload.model_fields_set:
            device.name = payload.name
        if "external_id" in payload.model_fields_set:
            device.external_id = payload.external_id
        session.add(device)
        session.flush()
        session.refresh(device)
        return device

    device = run_with_db_retry(_op, commit=True, operation_name="device_update")
    if not device:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    return DeviceResponse.model_validate(device)


@router.put(
    "/devices/{device_id}/user",
    response_model=DeviceResponse,
    dependencies=[Depends(require_admin)],
)
def link_device_user(
    device_id: uuid.UUID,
    payload: DeviceUserLink,
    tenant: TenantContext = Depends(get_tenant_context),
) -> DeviceResponse:
    def _op(session):
        device = (
            session.query(Device)
            .filter(Device.id == device_id)
            .filter(Device.tenant_id == tenant.id)
            .one_or_none()
        )
        if not device:
            return None, "device"
        user = (
            session.query(User)
            .filter(User.id == payload.user_id)
            .filter(User.tenant_id == tenant.id)
            .one_or_none()
        )
        if not user:
            return None, "user"
        device.user_id = user.id
        session.add(device)
        session.flush()
        session.refresh(device)
        return device, None

    device, missing = run_with_db_retry(_op, commit=True, operation_name="device_link_user")
    if missing == "device":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    if missing == "user":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return DeviceResponse.model_validate(device)


@router.delete(
    "/devices/{device_id}/user",
    response_model=DeviceResponse,
    dependencies=[Depends(require_admin)],
)
def unlink_device_user(
    device_id: uuid.UUID,
    tenant: TenantContext = Depends(get_tenant_context),
) -> DeviceResponse:
    def _op(session):
        device = (
            session.query(Device)
            .filter(Device.id == device_id)
            .filter(Device.tenant_id == tenant.id)
            .one_or_none()
        )
        if not device:
            return None
        device.user_id = None
        session.add(device)
        session.flush()
        session.refresh(device)
        return device

    device = run_with_db_retry(_op, commit=True, operation_name="device_unlink_user")
    if not device:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    return DeviceResponse.model_validate(device)
