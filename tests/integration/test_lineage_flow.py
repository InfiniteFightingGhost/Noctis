from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from tests.integration.utils import migrate_database


@pytest.mark.skipif(
    os.getenv("INTEGRATION_TEST_DATABASE_URL") is None,
    reason="Integration DB not configured",
)
@pytest.mark.anyio
async def test_snapshot_training_lineage_flow(tmp_path: Path) -> None:
    database_url = os.environ["INTEGRATION_TEST_DATABASE_URL"]
    os.environ["DATABASE_URL"] = database_url
    os.environ["PROMOTION_MIN_ACCURACY"] = "0"
    os.environ["PROMOTION_MIN_MACRO_F1"] = "0"
    model_registry = tmp_path / "models"
    model_registry.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path("models/active"), model_registry / "active")
    os.environ["MODEL_REGISTRY_PATH"] = str(model_registry)

    from app.core import settings as settings_module

    settings_module._get_settings_cached.cache_clear()

    migrate_database(database_url)

    from app.main import create_app
    from app.db.models import Prediction
    from app.db.session import run_with_db_retry
    from app.dataset.snapshot_config import DatasetSnapshotConfig
    from app.dataset.config import DatasetFilters, DatasetSplitConfig
    from app.dataset.snapshots import create_snapshot
    from app.experiments.service import register_model_version, register_training_run
    from app.promotion.service import promote_model
    from app.training.config import TrainingConfig
    from app.training.trainer import train_model
    from app.training.versioning import next_version
    from tests.utils.auth import build_auth_header, provision_service_client

    app = create_app()

    ingest_client = provision_service_client(role="ingest")
    read_client = provision_service_client(role="read")
    admin_client = provision_service_client(role="admin")
    ingest_headers = build_auth_header(ingest_client)
    read_headers = build_auth_header(read_client)
    admin_headers = build_auth_header(admin_client)

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
            for i in range(42):
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
            assert ingest["inserted"] == len(epochs)

            predict = (
                await client.post(
                    "/v1/predict",
                    headers=read_headers,
                    json={"recording_id": recording["id"], "epochs": epochs},
                )
            ).json()
            assert predict["predictions"]

    def _label_predictions(session):
        rows = (
            session.query(Prediction)
            .filter(Prediction.recording_id == uuid.UUID(recording["id"]))
            .order_by(Prediction.window_end_ts)
            .all()
        )
        labels = ["N2", "REM"]
        for idx, row in enumerate(rows):
            row.ground_truth_stage = labels[idx % len(labels)]
        return len(rows)

    updated = run_with_db_retry(_label_predictions, commit=True, operation_name="label_predictions")
    assert updated > 0

    snapshot_dir = tmp_path / "snapshot"
    snapshot_config = DatasetSnapshotConfig(
        name="integration_snapshot",
        output_dir=snapshot_dir,
        feature_schema_version="v1",
        window_size=21,
        allow_padding=False,
        label_strategy="ground_truth_only",
        balance_strategy="none",
        random_seed=42,
        export_format="npz",
        split=DatasetSplitConfig(),
        filters=DatasetFilters(),
    )
    snapshot_result = create_snapshot(snapshot_config)
    assert snapshot_result.row_count > 0

    training_config = TrainingConfig(
        dataset_dir=snapshot_dir,
        output_root=tmp_path / "trained_models",
        feature_schema_path=None,
        dataset_snapshot_id=str(snapshot_result.snapshot_id),
        model_type="gradient_boosting",
        random_seed=42,
        class_balance="none",
        feature_strategy="mean",
        hyperparameters={},
        evaluation_split_policy="none",
        version_bump="patch",
    )
    version = run_with_db_retry(
        lambda session: next_version(
            session=session,
            output_root=training_config.output_root,
            bump=training_config.version_bump,
        ),
        operation_name="next_model_version",
    )
    result = train_model(config=training_config, version=version)

    def _register(session):
        register_model_version(
            session,
            version=result.version,
            status="staging",
            metrics=result.metrics,
            feature_schema_version=result.feature_schema_version,
            artifact_path=str(result.artifact_dir),
            dataset_snapshot_id=result.dataset_snapshot_id,
            training_seed=result.training_seed,
            git_commit_hash=result.git_commit_hash,
            metrics_hash=result.metrics_hash,
            artifact_hash=result.artifact_hash,
        )
        register_training_run(
            session,
            experiment_id=None,
            model_version=result.version,
            status="completed",
            metrics=result.metrics,
            hyperparameters={"model_type": training_config.model_type},
            dataset_dir=str(training_config.dataset_dir),
            feature_schema_version=result.feature_schema_version,
            artifact_path=str(result.artifact_dir),
            dataset_snapshot_id=result.dataset_snapshot_id,
        )
        promote_model(
            session,
            version=result.version,
            actor="integration_test",
            reason="integration",
        )

    run_with_db_retry(_register, commit=True, operation_name="register_training")

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            reload_response = (await client.post("/v1/models/reload", headers=admin_headers)).json()
            assert reload_response["reloaded"] is True
