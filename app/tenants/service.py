from __future__ import annotations

import uuid

from sqlalchemy import func, select

from app.db.models import Tenant


def get_tenant_by_id(session, tenant_id: uuid.UUID) -> Tenant | None:
    return session.execute(
        select(Tenant).where(Tenant.id == tenant_id)
    ).scalar_one_or_none()


def count_active_tenants(session) -> int:
    return int(
        session.execute(
            select(func.count()).select_from(Tenant).where(Tenant.status == "active")
        ).scalar_one()
    )
