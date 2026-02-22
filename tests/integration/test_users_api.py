from __future__ import annotations

import os

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
async def test_create_user_and_link_device() -> None:
    database_url = os.environ["INTEGRATION_TEST_DATABASE_URL"]
    os.environ["DATABASE_URL"] = database_url
    _migrate(database_url)

    from app.main import create_app
    from tests.utils.auth import build_auth_header, provision_service_client

    admin_client = provision_service_client(role="admin")
    admin_headers = build_auth_header(admin_client)

    app = create_app()

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            user = (
                await client.post(
                    "/v1/users",
                    json={"name": "User One", "external_id": "user-1"},
                    headers=admin_headers,
                )
            ).json()

            device = (
                await client.post(
                    "/v1/devices",
                    json={"name": "device-1"},
                    headers=admin_headers,
                )
            ).json()

            linked = (
                await client.put(
                    f"/v1/devices/{device['id']}/user",
                    json={"user_id": user["id"]},
                    headers=admin_headers,
                )
            ).json()
            assert linked["user_id"] == user["id"]

            unlinked = (
                await client.delete(
                    f"/v1/devices/{device['id']}/user",
                    headers=admin_headers,
                )
            ).json()
            assert unlinked["user_id"] is None
