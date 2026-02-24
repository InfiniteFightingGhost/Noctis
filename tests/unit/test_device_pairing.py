from __future__ import annotations

from datetime import datetime, timedelta, timezone
import uuid

import pytest
from fastapi import HTTPException

from app.api import devices
from app.auth.context import AuthContext
from app.schemas.devices import (
    DeviceClaimByIdRequest,
    DevicePairingClaimRequest,
    DevicePairingStartRequest,
)
from app.tenants.context import TenantContext


class _Pairing:
    def __init__(self) -> None:
        self.id = uuid.uuid4()
        self.pairing_code = "ABC123"
        self.expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)


class _Device:
    def __init__(self) -> None:
        self.id = uuid.uuid4()
        self.name = "band"
        self.external_id = "ext-1"
        self.user_id = uuid.uuid4()
        self.created_at = datetime.now(timezone.utc)


def _user_auth() -> AuthContext:
    return AuthContext(
        client_id=uuid.uuid4(),
        client_name="user@example.com",
        role="user",
        tenant_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        scopes={"read", "ingest"},
        key_id="user-auth",
        principal_type="user",
    )


def _tenant() -> TenantContext:
    return TenantContext(
        id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        name="tenant",
        status="active",
    )


def test_pairing_start_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "run_with_db_retry", lambda *_args, **_kwargs: _Pairing())

    response = devices.start_device_pairing(
        DevicePairingStartRequest(device_external_id="ext-1"),
        _tenant(),
        _user_auth(),
    )

    assert response.pairing_code == "ABC123"


def test_pairing_claim_invalid_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "run_with_db_retry", lambda *_args, **_kwargs: (None, "invalid"))

    with pytest.raises(HTTPException) as exc:
        devices.claim_device_pairing(
            DevicePairingClaimRequest(pairing_session_id=uuid.uuid4(), pairing_code="BAD123"),
            _tenant(),
            _user_auth(),
        )

    assert exc.value.status_code == 404


def test_pairing_claim_expired_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "run_with_db_retry", lambda *_args, **_kwargs: (None, "expired"))

    with pytest.raises(HTTPException) as exc:
        devices.claim_device_pairing(
            DevicePairingClaimRequest(pairing_session_id=uuid.uuid4(), pairing_code="OLD123"),
            _tenant(),
            _user_auth(),
        )

    assert exc.value.status_code == 410


def test_pairing_claim_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "run_with_db_retry", lambda *_args, **_kwargs: (_Device(), None))

    response = devices.claim_device_pairing(
        DevicePairingClaimRequest(pairing_session_id=uuid.uuid4(), pairing_code="ABC123"),
        _tenant(),
        _user_auth(),
    )

    assert response.external_id == "ext-1"


def test_claim_device_by_id_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "run_with_db_retry", lambda *_args, **_kwargs: (_Device(), None))

    response = devices.claim_device_by_id(
        DeviceClaimByIdRequest(device_external_id="ext-1"),
        _tenant(),
        _user_auth(),
    )

    assert response.external_id == "ext-1"


def test_claim_device_by_id_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "run_with_db_retry", lambda *_args, **_kwargs: (None, "device"))

    with pytest.raises(HTTPException) as exc:
        devices.claim_device_by_id(
            DeviceClaimByIdRequest(device_external_id="missing"),
            _tenant(),
            _user_auth(),
        )

    assert exc.value.status_code == 404


def test_claim_device_by_id_rejects_already_paired(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "run_with_db_retry", lambda *_args, **_kwargs: (None, "bound"))

    with pytest.raises(HTTPException) as exc:
        devices.claim_device_by_id(
            DeviceClaimByIdRequest(device_external_id="ext-1"),
            _tenant(),
            _user_auth(),
        )

    assert exc.value.status_code == 409
