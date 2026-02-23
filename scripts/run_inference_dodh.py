from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import uuid

from sqlalchemy import select, desc
from sqlalchemy.dialects.postgresql import insert

from app.core.settings import get_settings
from typing import cast
from app.db.models import Device, Epoch, FeatureSchema, ModelVersion, Prediction, Recording
from app.db.session import run_with_db_retry
from app.ml.feature_schema import load_feature_schema
from app.ml.model import load_model
from app.ml.registry import LoadedModel
from app.services.inference import predict_windows
from app.services.windowing import WindowedEpoch, build_windows
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference for DODH recordings")
    parser.add_argument("--model-version", default=None, help="Model version to use")
    parser.add_argument("--device-external-id", default="dodh", help="Device external id")
    parser.add_argument("--limit", type=int, default=20, help="Max recordings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    tenant_id = uuid.UUID(settings.default_tenant_id)

    model_version = args.model_version or _latest_model_version()
    if not model_version:
        raise SystemExit("No model versions found")

    model_dir = Path(settings.model_registry_path) / str(model_version)
    if not model_dir.exists():
        raise SystemExit(f"Model dir not found: {model_dir}")

    bundle = load_model(model_dir)
    feature_schema = load_feature_schema(model_dir / "feature_schema.json")
    loaded_model = LoadedModel(
        version=str(model_version),
        model=bundle.model,
        feature_schema=feature_schema,
        metadata=bundle.metadata,
    )
    window_size = int(bundle.metadata.get("window_size") or 0)
    epoch_seconds = int(bundle.metadata.get("epoch_seconds") or 0)
    if window_size <= 0 or epoch_seconds <= 0:
        raise SystemExit("Model metadata missing window_size/epoch_seconds")

    device_id = _resolve_device_id(tenant_id, args.device_external_id)
    recordings = _list_recordings(tenant_id, device_id, limit=args.limit)
    if not recordings:
        raise SystemExit("No recordings found for device")

    dataset_snapshot_id = bundle.metadata.get("dataset_snapshot_id")
    if dataset_snapshot_id:
        try:
            dataset_snapshot_id = uuid.UUID(str(dataset_snapshot_id))
        except ValueError:
            dataset_snapshot_id = None

    for recording in recordings:
        epochs = _load_epochs(tenant_id, recording.id, feature_schema.version)
        if not epochs:
            continue
        schema_id = _resolve_schema_id(feature_schema.version)
        windowed_epochs = [
            WindowedEpoch(
                epoch_index=epoch.epoch_index,
                epoch_start_ts=epoch.epoch_start_ts,
                features=np.asarray(epoch.features_payload["features"], dtype=np.float32),
                feature_schema_id=schema_id,
            )
            for epoch in epochs
        ]
        windows = build_windows(
            windowed_epochs,
            window_size,
            allow_padding=False,
            epoch_seconds=epoch_seconds,
        )
        if not windows:
            continue
        predictions = predict_windows(loaded_model, [window.tensor for window in windows])
        ground_truth = _ground_truth_lookup(tenant_id, recording.id)
        rows = []
        for window, prediction in zip(windows, predictions, strict=True):
            label = str(prediction["predicted_stage"])
            confidence = float(cast(float, prediction.get("confidence", 0.0)))
            rows.append(
                {
                    "tenant_id": tenant_id,
                    "recording_id": recording.id,
                    "window_start_ts": window.start_ts,
                    "window_end_ts": window.end_ts,
                    "model_version": loaded_model.version,
                    "feature_schema_version": feature_schema.version,
                    "dataset_snapshot_id": dataset_snapshot_id,
                    "predicted_stage": label,
                    "ground_truth_stage": ground_truth.get(window.end_ts),
                    "confidence": confidence,
                    "probabilities": prediction["probabilities"],
                }
            )
        inserted = _insert_predictions(rows)
        print(f"recording={recording.id} windows={len(rows)} inserted={inserted}")


def _latest_model_version() -> str | None:
    def _op(session):
        row = session.execute(
            select(ModelVersion).order_by(desc(ModelVersion.created_at)).limit(1)
        ).scalar_one_or_none()
        return row.version if row else None

    return run_with_db_retry(_op, operation_name="latest_model_version")


def _resolve_device_id(tenant_id: uuid.UUID, external_id: str) -> uuid.UUID:
    def _op(session):
        device = (
            session.query(Device)
            .filter(Device.tenant_id == tenant_id)
            .filter(Device.external_id == external_id)
            .one_or_none()
        )
        if not device:
            raise ValueError("Device not found")
        return device.id

    return run_with_db_retry(_op, operation_name="resolve_device")


def _list_recordings(
    tenant_id: uuid.UUID,
    device_id: uuid.UUID,
    *,
    limit: int,
) -> list[Recording]:
    def _op(session):
        rows = (
            session.query(Recording)
            .filter(Recording.tenant_id == tenant_id)
            .filter(Recording.device_id == device_id)
            .order_by(Recording.started_at)
            .limit(limit)
            .all()
        )
        return rows

    return run_with_db_retry(_op, operation_name="list_recordings")


def _load_epochs(
    tenant_id: uuid.UUID,
    recording_id: uuid.UUID,
    schema_version: str,
) -> list[Epoch]:
    def _op(session):
        rows = (
            session.query(Epoch)
            .filter(Epoch.tenant_id == tenant_id)
            .filter(Epoch.recording_id == recording_id)
            .filter(Epoch.feature_schema_version == schema_version)
            .order_by(Epoch.epoch_index)
            .all()
        )
        return rows

    return run_with_db_retry(_op, operation_name="load_epochs")


def _ground_truth_lookup(
    tenant_id: uuid.UUID,
    recording_id: uuid.UUID,
) -> dict[datetime, str]:
    def _op(session):
        rows = (
            session.query(Prediction.window_end_ts, Prediction.ground_truth_stage)
            .filter(Prediction.tenant_id == tenant_id)
            .filter(Prediction.recording_id == recording_id)
            .filter(Prediction.model_version == "ground_truth")
            .all()
        )
        return {row[0]: row[1] for row in rows if row[1]}

    return run_with_db_retry(_op, operation_name="ground_truth_lookup")


def _insert_predictions(rows: list[dict]) -> int:
    if not rows:
        return 0

    def _op(session):
        stmt = insert(Prediction).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Prediction.recording_id, Prediction.window_end_ts],
            set_={
                "model_version": stmt.excluded.model_version,
                "predicted_stage": stmt.excluded.predicted_stage,
                "confidence": stmt.excluded.confidence,
                "probabilities": stmt.excluded.probabilities,
                "feature_schema_version": stmt.excluded.feature_schema_version,
                "dataset_snapshot_id": stmt.excluded.dataset_snapshot_id,
                "window_start_ts": stmt.excluded.window_start_ts,
            },
        )
        result = session.execute(stmt.returning(Prediction.id))
        return len(result.fetchall())

    return run_with_db_retry(_op, commit=True, operation_name="insert_predictions")


def _resolve_schema_id(schema_version: str) -> uuid.UUID:
    def _op(session):
        row = session.execute(
            select(FeatureSchema.id).where(FeatureSchema.version == schema_version)
        ).scalar_one_or_none()
        if row is None:
            raise ValueError("Feature schema not registered")
        return row

    return run_with_db_retry(_op, operation_name="resolve_schema_id")


if __name__ == "__main__":
    main()
