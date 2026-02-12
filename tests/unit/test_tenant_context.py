from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from app.auth.context import AuthContext
from app.tenants import context as tenant_context


class _Tenant:
    def __init__(self, tenant_id: uuid.UUID, status: str) -> None:
        self.id = tenant_id
        self.name = "tenant"
        self.status = status


def test_tenant_context_rejects_suspended(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id = uuid.uuid4()
    auth = AuthContext(
        client_id=uuid.uuid4(),
        client_name="svc",
        role="read",
        tenant_id=tenant_id,
        scopes={"read"},
        key_id="kid",
    )

    def fake_run_with_db_retry(_op, **_kwargs):
        return _Tenant(tenant_id, "suspended")

    monkeypatch.setattr(tenant_context, "run_with_db_retry", fake_run_with_db_retry)
    with pytest.raises(HTTPException):
        tenant_context.get_tenant_context(auth)
