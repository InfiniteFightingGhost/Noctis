from __future__ import annotations

from datetime import datetime, timedelta, timezone
import uuid

import jwt

from app.auth import service as auth_service


class _Client:
    def __init__(self, client_id: uuid.UUID, role: str, tenant_id: uuid.UUID) -> None:
        self.id = client_id
        self.name = "client"
        self.role = role
        self.tenant_id = tenant_id


class _Key:
    def __init__(self, key_id: str, secret: str) -> None:
        self.key_id = key_id
        self.secret = secret
        self.public_key = None


def test_authenticate_token_hs256(monkeypatch) -> None:
    client_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    key_id = uuid.uuid4().hex
    secret = "test-secret"
    client = _Client(client_id, "read", tenant_id)
    key = _Key(key_id, secret)

    def fake_run_with_db_retry(_op, **_kwargs):
        return (key, client)

    monkeypatch.setattr(auth_service, "run_with_db_retry", fake_run_with_db_retry)
    settings = auth_service.get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(client_id),
        "tenant_id": str(tenant_id),
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
    }
    token = jwt.encode(payload, secret, algorithm="HS256", headers={"kid": key_id})
    auth = auth_service.authenticate_token(token)
    assert auth.client_id == client_id
    assert auth.tenant_id == tenant_id
    assert auth.role == "read"
    assert "read" in auth.scopes
