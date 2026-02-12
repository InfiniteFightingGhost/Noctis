from __future__ import annotations

from datetime import datetime, timedelta, timezone
import uuid

import jwt

from app.core.settings import get_settings
from app.db.models import ServiceClient, ServiceClientKey, Tenant
from app.db.session import SessionLocal


def provision_service_client(
    *, role: str, tenant_id: str | None = None
) -> dict[str, str]:
    settings = get_settings()
    tenant_id = tenant_id or settings.default_tenant_id
    tenant_uuid = uuid.UUID(tenant_id)
    client_id = uuid.uuid4()
    key_id = uuid.uuid4().hex
    secret = uuid.uuid4().hex
    with SessionLocal() as session:
        tenant = session.get(Tenant, tenant_uuid)
        if tenant is None:
            session.add(
                Tenant(
                    id=tenant_uuid,
                    name="default",
                    status="active",
                )
            )
        client = ServiceClient(
            id=client_id,
            tenant_id=tenant_uuid,
            name=f"test-{client_id}",
            role=role,
            status="active",
        )
        key = ServiceClientKey(
            client_id=client_id,
            key_id=key_id,
            secret=secret,
            status="active",
        )
        session.add_all([client, key])
        session.commit()
    return {
        "client_id": str(client_id),
        "key_id": key_id,
        "secret": secret,
        "tenant_id": str(tenant_uuid),
    }


def build_auth_header(details: dict[str, str]) -> dict[str, str]:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": details["client_id"],
        "tenant_id": details["tenant_id"],
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=30)).timestamp()),
    }
    token = jwt.encode(
        payload,
        details["secret"],
        algorithm="HS256",
        headers={"kid": details["key_id"]},
    )
    return {"Authorization": f"Bearer {token}"}
