from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.metrics import INFERENCE_DURATION
from app.core.security import require_api_key
from app.core.settings import get_settings
from app.db.models import Epoch, Prediction, Recording
from app.db.session import get_db
from app.ml.feature_decode import decode_features
from app.schemas.predictions import PredictRequest, PredictResponse, PredictionItem
from app.services.inference import predict_windows
from app.services.windowing import WindowedEpoch, build_windows


router = APIRouter(tags=["predict"], dependencies=[Depends(require_api_key)])


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest, request: Request, db: Session = Depends(get_db)
) -> PredictResponse:
    settings = get_settings()
    recording = db.get(Recording, payload.recording_id)
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found"
        )

    registry = request.app.state.model_registry
    loaded_model = registry.get_loaded()

    if payload.epochs:
        epochs = payload.epochs
    else:
        epochs = (
            db.query(Epoch)
            .filter(Epoch.recording_id == payload.recording_id)
            .order_by(Epoch.epoch_index.desc())
            .limit(settings.window_size)
            .all()
        )
        epochs = list(reversed(epochs))

    if not epochs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No epochs available"
        )

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
        windowed_epochs.append(
            WindowedEpoch(
                epoch_index=epoch.epoch_index,
                epoch_start_ts=epoch.epoch_start_ts,
                features=vector,
            )
        )

    windows = build_windows(
        windowed_epochs, settings.window_size, settings.allow_window_padding
    )
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

    with INFERENCE_DURATION.time():
        predictions = predict_windows(
            loaded_model, [window.tensor for window in windows]
        )

    logging.getLogger("app").info(
        "predictions_generated",
        extra={
            "recording_id": str(payload.recording_id),
            "window_count": len(windows),
            "model_version": loaded_model.version,
        },
    )

    prediction_items: list[PredictionItem] = []
    for window, prediction in zip(windows, predictions, strict=True):
        prediction = prediction  # type: dict[str, Any]
        prediction_items.append(
            PredictionItem(
                window_start_ts=window.start_ts,
                window_end_ts=window.end_ts,
                predicted_stage=str(prediction["predicted_stage"]),
                confidence=float(prediction["confidence"]),
                probabilities=prediction["probabilities"],
            )
        )
        db.add(
            Prediction(
                recording_id=payload.recording_id,
                window_start_ts=window.start_ts,
                window_end_ts=window.end_ts,
                model_version=loaded_model.version,
                feature_schema_version=loaded_model.feature_schema.version,
                predicted_stage=str(prediction["predicted_stage"]),
                confidence=float(prediction["confidence"]),
                probabilities=prediction["probabilities"],
            )
        )
    db.commit()

    return PredictResponse(
        recording_id=payload.recording_id,
        model_version=loaded_model.version,
        feature_schema_version=loaded_model.feature_schema.version,
        predictions=prediction_items,
    )
