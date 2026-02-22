from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient


def _migrate(database_url: str) -> None:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")


@pytest.mark.skipif(
    os.getenv("INTEGRATION_TEST_DATABASE_URL") is None,
    reason="Integration DB not configured",
)
@pytest.mark.anyio
async def test_multitenant_isolation_and_auth() -> None:
    database_url = os.environ["INTEGRATION_TEST_DATABASE_URL"]
    os.environ["DATABASE_URL"] = database_url
    _migrate(database_url)

    from app.db.models import Tenant
    from app.db.session import SessionLocal
    from app.main import create_app
    from tests.utils.auth import build_auth_header, provision_service_client

    tenant_a = provision_service_client(role="ingest")
    tenant_b_id = str(uuid.uuid4())
    with SessionLocal() as session:
        session.add(
            Tenant(
                id=uuid.UUID(tenant_b_id),
                name="tenant-b",
                status="active",
            )
        )
        session.commit()
    tenant_b = provision_service_client(role="read", tenant_id=tenant_b_id)

    app = create_app()

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            device = (
                await client.post(
                    "/v1/devices",
                    json={"name": "device-1"},
                    headers=build_auth_header(tenant_a),
                )
            ).json()
            recording = (
                await client.post(
                    "/v1/recordings",
                    json={
                        "device_id": device["id"],
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    },
                    headers=build_auth_header(tenant_a),
                )
            ).json()

            cross_tenant = await client.get(
                f"/v1/recordings/{recording['id']}",
                headers=build_auth_header(tenant_b),
            )
            assert cross_tenant.status_code == 404

            missing_auth = await client.post("/v1/devices", json={"name": "device-2"})
            assert missing_auth.status_code == 401
