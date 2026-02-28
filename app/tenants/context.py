from __future__ import annotations

from dataclasses import dataclass
import uuid

from fastapi import Depends, HTTPException, status

from app.auth.context import AuthContext
from app.auth.dependencies import get_auth_context
from app.db.session import run_with_db_retry
from app.tenants.service import get_tenant_by_id


@dataclass(frozen=True)
class TenantContext:
    id: uuid.UUID
    name: str
    status: str


def get_tenant_context(auth: AuthContext = Depends(get_auth_context)) -> TenantContext:
    def _op(session):
        return get_tenant_by_id(session, auth.tenant_id)

    tenant = run_with_db_retry(_op, operation_name="get_tenant")
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tenant not found")
    if tenant.status != "active":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tenant is suspended")

    return TenantContext(id=tenant.id, name=tenant.name, status=tenant.status)
