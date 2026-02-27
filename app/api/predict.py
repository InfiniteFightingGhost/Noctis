from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.dialects.postgresql import insert

from app.auth.dependencies import require_scopes
from app.core.metrics import (
    INFERENCE_DURATION,
    PREDICTION_CONFIDENCE_HISTOGRAM,
    WINDOW_BUILD_DURATION,
)
from app.core.settings import get_settings
from app.db.models import (
    DatasetSnapshot,
    Epoch,
    FeatureStatistic,
    ModelUsageStat,
    Prediction,
    Recording,
)
from app.db.session import run_with_db_retry
from app.feature_store.service import get_feature_schema_by_version
from app.ml.feature_decode import decode_features
from app.schemas.predictions import PredictRequest, PredictResponse, PredictionItem
from app.monitoring.memory import memory_rss_mb
from app.monitoring.profiling import profile_block
from app.services.inference import predict_windows
from app.services.feature_stats import compute_daily_feature_stats, merge_feature_stats
from app.services.windowing import WindowedEpoch, build_windows
from app.tenants.context import TenantContext, get_tenant_context
from app.ml.validation import ensure_finite


router = APIRouter(tags=["predict"], dependencies=[Depends(require_scopes("read"))])


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    request: Request,
    batch_mode: bool | None = Query(default=None, alias="batch"),
    tenant: TenantContext = Depends(get_tenant_context),
) -> PredictResponse:
    settings = get_settings()

    def _recording(session):
        return (
            session.query(Recording)
            .filter(Recording.id == payload.recording_id)
            .filter(Recording.tenant_id == tenant.id)
            .one_or_none()
        )

    recording = run_with_db_retry(_recording, operation_name="predict_recording")
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")

    registry = request.app.state.model_registry
    loaded_model = registry.get_loaded()

    if payload.epochs:
        epochs = payload.epochs
    else:

        def _epochs(session):
            rows = (
                session.query(Epoch)
                .filter(Epoch.recording_id == payload.recording_id)
                .filter(Epoch.tenant_id == tenant.id)
                .order_by(Epoch.epoch_index.desc())
                .limit(settings.window_size)
                .all()
            )
            return list(reversed(rows))

        epochs = run_with_db_retry(_epochs, operation_name="predict_epochs")

    if not epochs:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No epochs available")

    schema_id = loaded_model.feature_schema.schema_id
    if schema_id is None:

        def _schema(session):
            schema = get_feature_schema_by_version(session, loaded_model.feature_schema.version)
            if schema is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Feature schema not registered",
                )
            return schema.id

        schema_id = run_with_db_retry(_schema, operation_name="predict_schema")

    windowed_epochs: list[WindowedEpoch] = []
    for epoch in epochs:
        schema_version = epoch.feature_schema_version
        if schema_version != loaded_model.feature_schema.version:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Feature schema mismatch",
            )
        if isinstance(epoch, Epoch):
            payload_features = epoch.features_payload["features"]
        else:
            payload_features = epoch.features
        vector = decode_features(payload_features, loaded_model.feature_schema)
        ensure_finite("features", vector)
        windowed_epochs.append(
            WindowedEpoch(
                epoch_index=epoch.epoch_index,
                epoch_start_ts=epoch.epoch_start_ts,
                features=vector,
                feature_schema_id=schema_id,
            )
        )

    window_build_start = time.perf_counter()
    windows = build_windows(
        windowed_epochs,
        settings.window_size,
        settings.allow_window_padding,
        epoch_seconds=settings.epoch_seconds,
    )
    WINDOW_BUILD_DURATION.observe(time.perf_counter() - window_build_start)
    if not windows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not enough contiguous epochs for prediction",
        )
    for window in windows:
        if window.tensor.shape != (
            settings.window_size,
            loaded_model.feature_schema.size,
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid window tensor shape",
            )

    predictions: list[dict[str, Any]] = []
    batch_size = settings.inference_batch_size
    inference_start = time.perf_counter()
    batch_enabled = settings.enable_batch_inference if batch_mode is None else batch_mode
    with (
        INFERENCE_DURATION.time(),
        profile_block("inference", window_count=len(windows)),
    ):
        if batch_enabled:
            for idx in range(0, len(windows), batch_size):
                batch_windows = [window.tensor for window in windows[idx : idx + batch_size]]
                predictions.extend(predict_windows(loaded_model, batch_windows))
        else:
            predictions = predict_windows(loaded_model, [window.tensor for window in windows])
    inference_duration_seconds = time.perf_counter() - inference_start

    logging.getLogger("app").info(
        "predictions_generated",
        extra={
            "recording_id": str(payload.recording_id),
            "tenant_id": str(tenant.id),
            "window_count": len(windows),
            "model_version": loaded_model.version,
            "memory_rss_mb": memory_rss_mb(),
        },
    )

    prediction_items: list[PredictionItem] = []
    prediction_rows: list[dict[str, Any]] = []
    dataset_snapshot_id = loaded_model.metadata.get("dataset_snapshot_id")
    if dataset_snapshot_id:
        try:
            dataset_snapshot_id = uuid.UUID(str(dataset_snapshot_id))
        except ValueError:
            dataset_snapshot_id = None
    if dataset_snapshot_id is not None:

        def _snapshot_exists(session) -> bool:
            return session.get(DatasetSnapshot, dataset_snapshot_id) is not None

        exists = run_with_db_retry(_snapshot_exists, operation_name="predict_snapshot_lookup")
        if not exists:
            dataset_snapshot_id = None
    daily_feature_stats = compute_daily_feature_stats(windowed_epochs)
    for window, prediction in zip(windows, predictions, strict=True):
        prediction_data: dict[str, Any] = prediction
        prediction_items.append(
            PredictionItem(
                window_start_ts=window.start_ts,
                window_end_ts=window.end_ts,
                predicted_stage=str(prediction_data["predicted_stage"]),
                confidence=float(prediction_data["confidence"]),
                probabilities=prediction_data["probabilities"],
            )
        )
        prediction_rows.append(
            {
                "tenant_id": tenant.id,
                "recording_id": payload.recording_id,
                "window_start_ts": window.start_ts,
                "window_end_ts": window.end_ts,
                "model_version": loaded_model.version,
                "feature_schema_version": loaded_model.feature_schema.version,
                "dataset_snapshot_id": dataset_snapshot_id,
                "predicted_stage": str(prediction_data["predicted_stage"]),
                "ground_truth_stage": None,
                "confidence": float(prediction_data["confidence"]),
                "probabilities": prediction_data["probabilities"],
            }
        )
        PREDICTION_CONFIDENCE_HISTOGRAM.observe(float(prediction_data["confidence"]))

    def _insert_predictions(session):
        inserted_count = 0
        if prediction_rows:
            stmt = insert(Prediction).values(prediction_rows)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=[Prediction.recording_id, Prediction.window_end_ts]
            )
            result = session.execute(stmt)
            inserted_count = int(result.rowcount or 0)
        if daily_feature_stats:
            _upsert_feature_stats(
                session,
                daily_feature_stats,
                tenant_id=tenant.id,
                recording_id=payload.recording_id,
                model_version=loaded_model.version,
                feature_schema_version=loaded_model.feature_schema.version,
            )
        if inserted_count > 0:
            session.add(
                ModelUsageStat(
                    tenant_id=tenant.id,
                    model_version=loaded_model.version,
                    window_start_ts=prediction_rows[0]["window_start_ts"],
                    window_end_ts=prediction_rows[-1]["window_end_ts"],
                    prediction_count=inserted_count,
                    average_latency_ms=(inference_duration_seconds * 1000.0 / inserted_count),
                )
            )

    run_with_db_retry(_insert_predictions, commit=True, operation_name="predict_insert")

    return PredictResponse(
        recording_id=payload.recording_id,
        model_version=loaded_model.version,
        feature_schema_version=loaded_model.feature_schema.version,
        predictions=prediction_items,
    )


def _upsert_feature_stats(
    session,
    daily_stats,
    *,
    tenant_id,
    recording_id,
    model_version,
    feature_schema_version,
) -> None:
    stat_dates = [item.stat_date for item in daily_stats]
    existing_rows = (
        session.query(FeatureStatistic)
        .filter(FeatureStatistic.tenant_id == tenant_id)
        .filter(FeatureStatistic.recording_id == recording_id)
        .filter(FeatureStatistic.model_version == model_version)
        .filter(FeatureStatistic.feature_schema_version == feature_schema_version)
        .filter(FeatureStatistic.stat_date.in_(stat_dates))
        .all()
    )
    existing_by_date = {row.stat_date: row for row in existing_rows}
    for aggregate in daily_stats:
        stat_date = aggregate.stat_date
        window_end_ts = datetime(
            stat_date.year,
            stat_date.month,
            stat_date.day,
            tzinfo=timezone.utc,
        )
        existing = existing_by_date.get(stat_date)
        if existing:
            existing.stats = merge_feature_stats(existing.stats, aggregate.stats)
            existing.window_end_ts = window_end_ts
        else:
            session.add(
                FeatureStatistic(
                    tenant_id=tenant_id,
                    recording_id=recording_id,
                    model_version=model_version,
                    feature_schema_version=feature_schema_version,
                    stat_date=stat_date,
                    window_end_ts=window_end_ts,
                    stats=aggregate.stats,
                )
            )
