from __future__ import annotations

from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.models import Device, Epoch, Prediction, Recording
from app.db.session import get_db
from app.schemas.epochs import EpochResponse
from app.schemas.predictions import PredictionResponse
from app.schemas.recordings import RecordingCreate, RecordingResponse
from app.schemas.summary import RecordingSummary
from app.services.summary import summarize_predictions


router = APIRouter(tags=["recordings"])


@router.post("/recordings", response_model=RecordingResponse)
def create_recording(
    payload: RecordingCreate, db: Session = Depends(get_db)
) -> RecordingResponse:
    device = db.get(Device, payload.device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Device not found"
        )
    recording = Recording(
        device_id=payload.device_id,
        started_at=payload.started_at,
        timezone=payload.timezone,
    )
    db.add(recording)
    db.commit()
    db.refresh(recording)
    return RecordingResponse.model_validate(recording)


@router.get("/recordings/{recording_id}", response_model=RecordingResponse)
def get_recording(
    recording_id: uuid.UUID, db: Session = Depends(get_db)
) -> RecordingResponse:
    recording = db.get(Recording, recording_id)
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found"
        )
    return RecordingResponse.model_validate(recording)


@router.get("/recordings/{recording_id}/epochs", response_model=list[EpochResponse])
def get_epochs(
    recording_id: uuid.UUID,
    from_ts: datetime = Query(..., alias="from"),
    to_ts: datetime = Query(..., alias="to"),
    db: Session = Depends(get_db),
) -> list[EpochResponse]:
    rows = (
        db.query(Epoch)
        .filter(Epoch.recording_id == recording_id)
        .filter(Epoch.epoch_start_ts >= from_ts, Epoch.epoch_start_ts <= to_ts)
        .order_by(Epoch.epoch_index)
        .all()
    )
    return [EpochResponse.model_validate(row) for row in rows]


@router.get(
    "/recordings/{recording_id}/predictions", response_model=list[PredictionResponse]
)
def get_predictions(
    recording_id: uuid.UUID,
    from_ts: datetime = Query(..., alias="from"),
    to_ts: datetime = Query(..., alias="to"),
    db: Session = Depends(get_db),
) -> list[PredictionResponse]:
    rows = (
        db.query(Prediction)
        .filter(Prediction.recording_id == recording_id)
        .filter(Prediction.window_end_ts >= from_ts, Prediction.window_end_ts <= to_ts)
        .order_by(Prediction.window_end_ts)
        .all()
    )
    return [PredictionResponse.model_validate(row) for row in rows]


@router.get("/recordings/{recording_id}/summary", response_model=RecordingSummary)
def get_summary(
    recording_id: uuid.UUID,
    from_ts: datetime = Query(..., alias="from"),
    to_ts: datetime = Query(..., alias="to"),
    db: Session = Depends(get_db),
) -> RecordingSummary:
    stages = (
        db.query(Prediction.predicted_stage)
        .filter(Prediction.recording_id == recording_id)
        .filter(Prediction.window_end_ts >= from_ts, Prediction.window_end_ts <= to_ts)
        .order_by(Prediction.window_end_ts)
        .all()
    )
    stage_list = [row[0] for row in stages]
    return summarize_predictions(str(recording_id), from_ts, to_ts, stage_list)
