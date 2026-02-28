from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from tests.integration.utils import migrate_database


@pytest.mark.skipif(
    os.getenv("INTEGRATION_TEST_DATABASE_URL") is None,
    reason="Integration DB not configured",
)
@pytest.mark.anyio
async def test_workflow_support_endpoints() -> None:
    database_url = os.environ["INTEGRATION_TEST_DATABASE_URL"]
    os.environ["DATABASE_URL"] = database_url
    migrate_database(database_url)

    from app.main import create_app
    from tests.utils.auth import build_auth_header, provision_service_client

    app = create_app()
    ingest_client = provision_service_client(role="ingest")
    read_client = provision_service_client(role="read")
    ingest_headers = build_auth_header(ingest_client)
    read_headers = build_auth_header(read_client)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            device = (
                await client.post("/v1/devices", json={"name": "device-1"}, headers=ingest_headers)
            ).json()
            recording = (
                await client.post(
                    "/v1/recordings",
                    json={
                        "device_id": device["id"],
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    },
                    headers=ingest_headers,
                )
            ).json()

            start = datetime.now(timezone.utc)
            epochs = [
                {
                    "epoch_index": index,
                    "epoch_start_ts": (start + timedelta(seconds=30 * index)).isoformat(),
                    "feature_schema_version": "v1",
                    "features": [0.1] * 10,
                }
                for index in range(21)
            ]
            ingest = await client.post(
                "/v1/epochs:ingest",
                headers=ingest_headers,
                json={"recording_id": recording["id"], "epochs": epochs},
            )
            assert ingest.status_code == 200

            predict = await client.post(
                "/v1/predict",
                headers=read_headers,
                json={"recording_id": recording["id"]},
            )
            assert predict.status_code == 200

            alarm = await client.get("/v1/alarm", headers=read_headers)
            assert alarm.status_code == 200
            assert "wake_time" in alarm.json()

            routine = await client.get("/v1/routines/current", headers=read_headers)
            assert routine.status_code == 200
            assert "steps" in routine.json()

            summary = await client.get("/v1/sleep/latest/summary", headers=read_headers)
            assert summary.status_code == 200
            assert summary.json()["recordingId"] == recording["id"]
