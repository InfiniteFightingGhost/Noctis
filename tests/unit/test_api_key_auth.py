from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.requests import Request

from app.auth import api_key
from app.auth.service import AuthError


def _request(path: str, *, headers: dict[str, str] | None = None) -> Request:
    encoded_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": encoded_headers,
    }
    return Request(scope)


def test_authenticate_hardware_api_key_accepts_ingest_device_path(monkeypatch) -> None:
    monkeypatch.setattr(
        api_key,
        "get_settings",
        lambda: SimpleNamespace(
            api_v1_prefix="/v1",
            api_key_header="X-API-Key",
            api_key="esp32-shared-secret",
            default_tenant_id="00000000-0000-0000-0000-000000000001",
        ),
    )
    request = _request(
        "/v1/epochs:ingest-device",
        headers={"X-API-Key": "esp32-shared-secret"},
    )

    auth = api_key.authenticate_hardware_api_key(request)

    assert auth is not None
    assert auth.role == "ingest"
    assert auth.principal_type == "service"
    assert auth.key_id == "hardware-api-key"
    assert "ingest" in auth.scopes
    assert "read" in auth.scopes


@pytest.mark.parametrize(
    "path",
    [
        "/v1/recordings:start",
        "/v1/sleep/latest/summary",
        "/v1/recordings",
        "/v1/recordings/550e8400-e29b-41d4-a716-446655440000",
        "/v1/recordings/550e8400-e29b-41d4-a716-446655440000/epochs",
        "/v1/recordings/550e8400-e29b-41d4-a716-446655440000/predictions",
    ],
)
def test_authenticate_hardware_api_key_accepts_whitelisted_paths(monkeypatch, path) -> None:
    monkeypatch.setattr(
        api_key,
        "get_settings",
        lambda: SimpleNamespace(
            api_v1_prefix="/v1",
            api_key_header="X-API-Key",
            api_key="esp32-shared-secret",
            default_tenant_id="00000000-0000-0000-0000-000000000001",
        ),
    )
    request = _request(path, headers={"X-API-Key": "esp32-shared-secret"})

    auth = api_key.authenticate_hardware_api_key(request)

    assert auth is not None
    assert auth.role == "ingest"


def test_authenticate_hardware_api_key_ignores_other_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        api_key,
        "get_settings",
        lambda: SimpleNamespace(
            api_v1_prefix="/v1",
            api_key_header="X-API-Key",
            api_key="esp32-shared-secret",
            default_tenant_id="00000000-0000-0000-0000-000000000001",
        ),
    )
    request = _request("/v1/epochs:ingest", headers={"X-API-Key": "esp32-shared-secret"})

    auth = api_key.authenticate_hardware_api_key(request)

    assert auth is None


def test_authenticate_hardware_api_key_rejects_invalid_key(monkeypatch) -> None:
    monkeypatch.setattr(
        api_key,
        "get_settings",
        lambda: SimpleNamespace(
            api_v1_prefix="/v1",
            api_key_header="X-API-Key",
            api_key="esp32-shared-secret",
            default_tenant_id="00000000-0000-0000-0000-000000000001",
        ),
    )
    request = _request(
        "/v1/epochs:ingest-device",
        headers={"X-API-Key": "wrong-key"},
    )

    with pytest.raises(AuthError) as exc:
        api_key.authenticate_hardware_api_key(request)

    assert exc.value.code == "invalid_api_key"
