from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

from tests.integration.utils import migrate_database


@pytest.mark.skipif(
    os.getenv("INTEGRATION_TEST_DATABASE_URL") is None,
    reason="Integration DB not configured",
)
@pytest.mark.anyio
async def test_timescale_policy_apply() -> None:
    database_url = os.environ["INTEGRATION_TEST_DATABASE_URL"]
    os.environ["DATABASE_URL"] = database_url
    migrate_database(database_url)

    from app.main import create_app
    from tests.utils.auth import build_auth_header, provision_service_client

    app = create_app()
    admin = provision_service_client(role="admin")
    headers = build_auth_header(admin)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            dry_run = await client.post("/internal/timescale/dry-run", headers=headers)
            assert dry_run.status_code == 200
            apply = await client.post("/internal/timescale/apply", headers=headers)
            assert apply.status_code == 200
