from __future__ import annotations

from datetime import datetime, timedelta, timezone
import secrets
import string
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.context import AuthContext
from app.auth.dependencies import get_auth_context, require_admin, require_scopes
from app.db.models import Device, DevicePairingSession, User
from app.db.session import run_with_db_retry
from app.schemas.devices import (
    DeviceClaimByIdRequest,
    DeviceCreate,
    DevicePairingClaimRequest,
    DevicePairingStartRequest,
    DevicePairingStartResponse,
    DeviceResponse,
    DeviceUpdate,
    DeviceUserLink,
)
from app.services.user_identity import resolve_or_create_domain_user_for_auth
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["devices"])

PAIRING_CODE_LENGTH = 6
PAIRING_TTL_MINUTES = 10


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


@router.post(
    "/devices/claim-by-id",
    response_model=DeviceResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def claim_device_by_id(
    payload: DeviceClaimByIdRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    auth: AuthContext = Depends(get_auth_context),
) -> DeviceResponse:
    if auth.principal_type != "user":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Device claim requires user principal",
        )
    external_id = payload.device_external_id.strip()
    if not external_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="device_external_id is required",
        )

    def _op(session):
        user = resolve_or_create_domain_user_for_auth(session, tenant_id=tenant.id, auth=auth)
        device = (
            session.query(Device)
            .filter(Device.tenant_id == tenant.id)
            .filter(Device.external_id == external_id)
            .one_or_none()
        )
        if device is None:
            return None, "device"
        if device.user_id is not None and device.user_id != user.id:
            return None, "bound"
        device.user_id = user.id
        session.add(device)
        session.flush()
        session.refresh(device)
        return device, None

    device, error = run_with_db_retry(_op, commit=True, operation_name="device_claim_by_id")
    if error == "device":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    if error == "bound":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Device already paired")
    return DeviceResponse.model_validate(device)


@router.post(
    "/devices/pairing/start",
    response_model=DevicePairingStartResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def start_device_pairing(
    payload: DevicePairingStartRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    auth: AuthContext = Depends(get_auth_context),
) -> DevicePairingStartResponse:
    if auth.principal_type != "user":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Pairing requires user principal",
        )
    if payload.device_id is None and not payload.device_external_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="device_id or device_external_id is required",
        )

    def _op(session):
        user = resolve_or_create_domain_user_for_auth(session, tenant_id=tenant.id, auth=auth)
        query = session.query(Device).filter(Device.tenant_id == tenant.id)
        if payload.device_id is not None:
            query = query.filter(Device.id == payload.device_id)
        else:
            query = query.filter(Device.external_id == payload.device_external_id)
        device = query.one_or_none()
        if device is None:
            return None
        pairing = DevicePairingSession(
            tenant_id=tenant.id,
            device_id=device.id,
            created_by_user_id=user.id,
            pairing_code=_generate_pairing_code(),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=PAIRING_TTL_MINUTES),
        )
        session.add(pairing)
        session.flush()
        session.refresh(pairing)
        return pairing

    pairing = run_with_db_retry(_op, commit=True, operation_name="device_pairing_start")
    if pairing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    return DevicePairingStartResponse(
        pairing_session_id=pairing.id,
        pairing_code=pairing.pairing_code,
        expires_at=pairing.expires_at,
    )


@router.post(
    "/devices/pairing/claim",
    response_model=DeviceResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def claim_device_pairing(
    payload: DevicePairingClaimRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    auth: AuthContext = Depends(get_auth_context),
) -> DeviceResponse:
    if auth.principal_type != "user":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Pairing requires user principal",
        )

    def _op(session):
        user = resolve_or_create_domain_user_for_auth(session, tenant_id=tenant.id, auth=auth)
        pairing = (
            session.query(DevicePairingSession)
            .filter(DevicePairingSession.id == payload.pairing_session_id)
            .filter(DevicePairingSession.tenant_id == tenant.id)
            .filter(DevicePairingSession.pairing_code == payload.pairing_code.strip().upper())
            .one_or_none()
        )
        if pairing is None:
            return None, "invalid"
        if pairing.claimed_at is not None:
            return None, "claimed"
        if pairing.expires_at < datetime.now(timezone.utc):
            return None, "expired"
        device = (
            session.query(Device)
            .filter(Device.id == pairing.device_id)
            .filter(Device.tenant_id == tenant.id)
            .one_or_none()
        )
        if device is None:
            return None, "device"
        if device.user_id is not None and device.user_id != user.id:
            return None, "bound"
        pairing.claimed_at = datetime.now(timezone.utc)
        pairing.claimed_by_user_id = user.id
        device.user_id = user.id
        session.add(pairing)
        session.add(device)
        session.flush()
        session.refresh(device)
        return device, None

    device, error = run_with_db_retry(_op, commit=True, operation_name="device_pairing_claim")
    if error == "invalid":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Pairing session not found"
        )
    if error == "claimed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Pairing session already claimed"
        )
    if error == "expired":
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Pairing session expired")
    if error == "device":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    if error == "bound":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Device already paired")
    return DeviceResponse.model_validate(device)


def _generate_pairing_code() -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(PAIRING_CODE_LENGTH))
