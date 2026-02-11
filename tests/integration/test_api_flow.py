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
    os.environ["API_KEY"] = "test-key"
    _migrate(database_url)

    from app.main import create_app

    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        device = (await client.post("/v1/devices", json={"name": "device-1"})).json()
        recording = (
            await client.post(
                "/v1/recordings",
                json={
                    "device_id": device["id"],
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
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
                headers={"X-API-Key": "test-key"},
                json={"recording_id": recording["id"], "epochs": epochs},
            )
        ).json()
        assert ingest["inserted"] == 21

        predict = (
            await client.post(
                "/v1/predict",
                headers={"X-API-Key": "test-key"},
                json={"recording_id": recording["id"]},
            )
        ).json()
        assert predict["model_version"] == "active"
        assert len(predict["predictions"]) == 1
