from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

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
async def test_ingest_predict_flow() -> None:
    database_url = os.environ["INTEGRATION_TEST_DATABASE_URL"]
    os.environ["DATABASE_URL"] = database_url
    _migrate(database_url)

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
                await client.post(
                    "/v1/devices",
                    json={"name": "device-1"},
                    headers=ingest_headers,
                )
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
            epochs = []
            for i in range(21):
                epochs.append(
                    {
                        "epoch_index": i,
                        "epoch_start_ts": (start + timedelta(seconds=30 * i)).isoformat(),
                        "feature_schema_version": "v1",
                        "features": [0.1] * 10,
                    }
                )

            ingest = (
                await client.post(
                    "/v1/epochs:ingest",
                    headers=ingest_headers,
                    json={"recording_id": recording["id"], "epochs": epochs},
                )
            ).json()
            assert ingest["inserted"] == 21

            predict = (
                await client.post(
                    "/v1/predict",
                    headers=read_headers,
                    json={"recording_id": recording["id"]},
                )
            ).json()
            assert predict["model_version"] == "active"
            assert len(predict["predictions"]) == 1

            evaluation = (
                await client.get(
                    f"/v1/recordings/{recording['id']}/evaluation",
                    headers=read_headers,
                    params={
                        "from": start.isoformat(),
                        "to": (start + timedelta(minutes=11)).isoformat(),
                    },
                )
            ).json()
            assert evaluation["scope"] == "recording"
            assert evaluation["total_predictions"] >= 1

            drift = (await client.get("/v1/model/drift", headers=read_headers)).json()
            assert "metrics" in drift
