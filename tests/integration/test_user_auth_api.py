from __future__ import annotations

import os
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from tests.integration.utils import migrate_database


@pytest.mark.skipif(
    os.getenv("INTEGRATION_TEST_DATABASE_URL") is None,
    reason="Integration DB not configured",
)
@pytest.mark.anyio
async def test_user_auth_flow() -> None:
    database_url = os.environ["INTEGRATION_TEST_DATABASE_URL"]
    os.environ["DATABASE_URL"] = database_url
    migrate_database(database_url)

    from app.main import create_app

    app = create_app()
    username = f"user_{uuid.uuid4().hex[:8]}"
    email = f"user-{uuid.uuid4().hex}@example.com"
    password = "Str0ngPassw0rd!"

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            register_response = await client.post(
                "/v1/auth/register",
                json={"username": username, "email": email, "password": password},
            )
            assert register_response.status_code == 201
            register_payload = register_response.json()
            assert register_payload["user"]["username"] == username
            assert register_payload["user"]["email"] == email
            assert register_payload["access_token"]

            duplicate_response = await client.post(
                "/v1/auth/register",
                json={"username": username, "email": email, "password": password},
            )
            assert duplicate_response.status_code == 409

            login_response = await client.post(
                "/v1/auth/login",
                json={"email": email, "password": password},
            )
            assert login_response.status_code == 200
            login_payload = login_response.json()
            assert login_payload["user"]["username"] == username
            assert login_payload["user"]["email"] == email
            token = login_payload["access_token"]

            invalid_password = await client.post(
                "/v1/auth/login",
                json={"email": email, "password": "wrongpassword"},
            )
            assert invalid_password.status_code == 401
            assert invalid_password.json()["error"]["message"] == "Invalid credentials"

            no_token = await client.get("/v1/auth/me")
            assert no_token.status_code == 401

            invalid_token = await client.get(
                "/v1/auth/me",
                headers={"Authorization": "Bearer invalid-token"},
            )
            assert invalid_token.status_code == 401

            me_response = await client.get(
                "/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert me_response.status_code == 200
            assert me_response.json()["username"] == username
            assert me_response.json()["email"] == email
