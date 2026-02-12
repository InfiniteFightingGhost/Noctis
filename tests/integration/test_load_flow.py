from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient

from app.db.models import FeatureStatistic, Prediction


def _migrate(database_url: str) -> None:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")


@pytest.mark.skipif(
    os.getenv("INTEGRATION_TEST_DATABASE_URL") is None,
    reason="Integration DB not configured",
)
@pytest.mark.anyio
async def test_load_and_drift_flow() -> None:
    database_url = os.environ["INTEGRATION_TEST_DATABASE_URL"]
    os.environ["DATABASE_URL"] = database_url
    os.environ["PERFORMANCE_SAMPLE_SIZE"] = "2"
    _migrate(database_url)

    from app.main import create_app
    from tests.utils.auth import build_auth_header, provision_service_client

    app = create_app()

    ingest_client = provision_service_client(role="ingest")
    read_client = provision_service_client(role="read")
    admin_client = provision_service_client(role="admin")
    ingest_headers = build_auth_header(ingest_client)
    read_headers = build_auth_header(read_client)
    admin_headers = build_auth_header(admin_client)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        recordings: list[str] = []
        for idx in range(100):
            device = (
                await client.post(
                    "/v1/devices",
                    json={"name": f"device-{idx}"},
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
            recordings.append(recording["id"])

        start = datetime.now(timezone.utc)
        epochs_payload = []
        for i in range(21):
            epochs_payload.append(
                {
                    "epoch_index": i,
                    "epoch_start_ts": (start + timedelta(seconds=30 * i)).isoformat(),
                    "feature_schema_version": "v1",
                    "features": [0.1] * 10,
                }
            )

        for recording_id in recordings:
            ingest = (
                await client.post(
                    "/v1/epochs:ingest",
                    headers=ingest_headers,
                    json={"recording_id": recording_id, "epochs": epochs_payload},
                )
            ).json()
            assert ingest["inserted"] == 21

        async def _predict(recording_id: str):
            return await client.post(
                "/v1/predict",
                headers=read_headers,
                json={"recording_id": recording_id},
            )

        tasks = [_predict(recordings[idx]) for idx in range(10)]
        reload_task = client.post("/v1/models/reload", headers=admin_headers)
        results = await asyncio.gather(*tasks, reload_task)
        for response in results[:-1]:
            assert response.status_code == 200
        reload_response = results[-1]
        assert reload_response.status_code == 200
        payload = reload_response.json()
        assert payload["reloaded"] is True
        assert payload["model_version"] == "active"

        _seed_drift_data(recordings[0], ingest_client["tenant_id"])
        drift = (await client.get("/v1/model/drift", headers=read_headers)).json()
        assert drift["metrics"]
        assert drift["flagged_features"]


def _seed_drift_data(recording_id: str, tenant_id: str) -> None:
    from app.db.session import SessionLocal

    session = SessionLocal()
    try:
        start = datetime.now(timezone.utc)
        rows = []
        for idx in range(4):
            stage = "W" if idx < 2 else "N3"
            rows.append(
                Prediction(
                    tenant_id=tenant_id,
                    recording_id=recording_id,
                    window_start_ts=start + timedelta(seconds=30 * idx),
                    window_end_ts=start + timedelta(seconds=30 * (idx + 1)),
                    model_version="active",
                    feature_schema_version="v1",
                    predicted_stage=stage,
                    ground_truth_stage=None,
                    probabilities={stage: 1.0},
                    confidence=0.99,
                )
            )
        session.add_all(rows)
        today = datetime.now(timezone.utc).date()
        session.query(FeatureStatistic).filter(
            FeatureStatistic.tenant_id == tenant_id,
            FeatureStatistic.recording_id == recording_id,
            FeatureStatistic.model_version == "active",
            FeatureStatistic.feature_schema_version == "v1",
            FeatureStatistic.stat_date.in_(
                [
                    today - timedelta(days=3),
                    today - timedelta(days=2),
                    today - timedelta(days=1),
                    today,
                ]
            ),
        ).delete(synchronize_session=False)
        stats_rows = [
            FeatureStatistic(
                tenant_id=tenant_id,
                recording_id=recording_id,
                model_version="active",
                feature_schema_version="v1",
                stat_date=today - timedelta(days=3),
                window_end_ts=datetime.combine(
                    today - timedelta(days=3), datetime.min.time(), tzinfo=timezone.utc
                ),
                stats={"means": [0.0, 0.0], "count": 2},
            ),
            FeatureStatistic(
                tenant_id=tenant_id,
                recording_id=recording_id,
                model_version="active",
                feature_schema_version="v1",
                stat_date=today - timedelta(days=2),
                window_end_ts=datetime.combine(
                    today - timedelta(days=2), datetime.min.time(), tzinfo=timezone.utc
                ),
                stats={"means": [0.2, 0.1], "count": 2},
            ),
            FeatureStatistic(
                tenant_id=tenant_id,
                recording_id=recording_id,
                model_version="active",
                feature_schema_version="v1",
                stat_date=today - timedelta(days=1),
                window_end_ts=datetime.combine(
                    today - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
                ),
                stats={"means": [10.0, 9.0], "count": 2},
            ),
            FeatureStatistic(
                tenant_id=tenant_id,
                recording_id=recording_id,
                model_version="active",
                feature_schema_version="v1",
                stat_date=today,
                window_end_ts=datetime.combine(
                    today, datetime.min.time(), tzinfo=timezone.utc
                ),
                stats={"means": [12.0, 8.5], "count": 2},
            ),
        ]
        session.add_all(stats_rows)
        session.commit()
    finally:
        session.close()
