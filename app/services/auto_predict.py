from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from sqlalchemy.dialects.postgresql import insert

from app.core.settings import get_settings
from app.db.models import Epoch, FeatureStatistic, ModelUsageStat, Prediction
from app.db.session import run_with_db_retry
from app.feature_store.service import get_feature_schema_by_version
from app.ml.feature_decode import decode_features
from app.ml.validation import ensure_finite
from app.services.feature_stats import compute_daily_feature_stats, merge_feature_stats
from app.services.inference import predict_windows
from app.services.windowing import WindowedEpoch, build_windows


def auto_predict_recording(*, tenant_id: uuid.UUID, recording_id: uuid.UUID, registry) -> None:
    settings = get_settings()
    try:
        loaded_model = registry.get_loaded()
    except Exception:
        logging.getLogger("app").exception("auto_predict_model_not_ready")
        return

    def _epochs(session):
        rows = (
            session.query(Epoch)
            .filter(Epoch.recording_id == recording_id)
            .filter(Epoch.tenant_id == tenant_id)
            .order_by(Epoch.epoch_index.desc())
            .limit(settings.window_size)
            .all()
        )
        return list(reversed(rows))

    epochs = run_with_db_retry(_epochs, operation_name="auto_predict_epochs")
    if not epochs:
        return

    schema_id = loaded_model.feature_schema.schema_id
    if schema_id is None:

        def _schema(session):
            schema = get_feature_schema_by_version(session, loaded_model.feature_schema.version)
            if schema is None:
                return None
            return schema.id

        schema_id = run_with_db_retry(_schema, operation_name="auto_predict_schema")
        if schema_id is None:
            return

    windowed_epochs: list[WindowedEpoch] = []
    for epoch in epochs:
        schema_version = epoch.feature_schema_version
        if schema_version != loaded_model.feature_schema.version:
            return
        payload_features = epoch.features_payload["features"]
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

    windows = build_windows(
        windowed_epochs,
        settings.window_size,
        settings.allow_window_padding,
        epoch_seconds=settings.epoch_seconds,
    )
    if not windows:
        return

    inference_start = time.perf_counter()
    predictions = predict_windows(loaded_model, [window.tensor for window in windows])
    inference_duration_seconds = time.perf_counter() - inference_start

    prediction_rows = []
    for window, prediction in zip(windows, predictions, strict=True):
        prediction_rows.append(
            {
                "tenant_id": tenant_id,
                "recording_id": recording_id,
                "window_start_ts": window.start_ts,
                "window_end_ts": window.end_ts,
                "model_version": loaded_model.version,
                "feature_schema_version": loaded_model.feature_schema.version,
                "dataset_snapshot_id": None,
                "predicted_stage": str(prediction["predicted_stage"]),
                "ground_truth_stage": None,
                "confidence": float(prediction["confidence"]),
                "probabilities": prediction["probabilities"],
            }
        )

    daily_feature_stats = compute_daily_feature_stats(windowed_epochs)

    def _persist(session):
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
                tenant_id=tenant_id,
                recording_id=recording_id,
                model_version=loaded_model.version,
                feature_schema_version=loaded_model.feature_schema.version,
            )
        if inserted_count > 0:
            session.add(
                ModelUsageStat(
                    tenant_id=tenant_id,
                    model_version=loaded_model.version,
                    window_start_ts=prediction_rows[0]["window_start_ts"],
                    window_end_ts=prediction_rows[-1]["window_end_ts"],
                    prediction_count=inserted_count,
                    average_latency_ms=(inference_duration_seconds * 1000.0 / inserted_count),
                )
            )

    try:
        run_with_db_retry(_persist, commit=True, operation_name="auto_predict_insert")
    except Exception:
        logging.getLogger("app").exception("auto_predict_persist_failed")


def _upsert_feature_stats(
    session,
    daily_stats,
    *,
    tenant_id,
    recording_id,
    model_version,
    feature_schema_version,
):
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
