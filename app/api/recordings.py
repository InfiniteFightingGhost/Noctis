from __future__ import annotations

from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.auth.dependencies import require_scopes
from app.db.models import Device, Epoch, Prediction, Recording
from app.db.session import run_with_db_retry
from app.schemas.epochs import EpochResponse
from app.schemas.predictions import PredictionResponse
from app.schemas.recordings import RecordingCreate, RecordingResponse
from app.schemas.summary import RecordingSummary
from app.services.summary import summarize_predictions
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["recordings"])


@router.post(
    "/recordings",
    response_model=RecordingResponse,
    dependencies=[Depends(require_scopes("ingest"))],
)
def create_recording(
    payload: RecordingCreate,
    tenant: TenantContext = Depends(get_tenant_context),
) -> RecordingResponse:
    def _device(session):
        return (
            session.query(Device)
            .filter(Device.id == payload.device_id)
            .filter(Device.tenant_id == tenant.id)
            .one_or_none()
        )

    device = run_with_db_retry(_device, operation_name="recording_device")
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Device not found"
        )

    def _create(session):
        recording = Recording(
            tenant_id=tenant.id,
            device_id=payload.device_id,
            started_at=payload.started_at,
            timezone=payload.timezone,
        )
        session.add(recording)
        session.flush()
        session.refresh(recording)
        return recording

    recording = run_with_db_retry(
        _create, commit=True, operation_name="recording_create"
    )
    return RecordingResponse.model_validate(recording)


@router.get(
    "/recordings/{recording_id}",
    response_model=RecordingResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def get_recording(
    recording_id: uuid.UUID,
    tenant: TenantContext = Depends(get_tenant_context),
) -> RecordingResponse:
    def _op(session):
        return (
            session.query(Recording)
            .filter(Recording.id == recording_id)
            .filter(Recording.tenant_id == tenant.id)
            .one_or_none()
        )

    recording = run_with_db_retry(_op, operation_name="recording_get")
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found"
        )
    return RecordingResponse.model_validate(recording)


@router.get(
    "/recordings/{recording_id}/epochs",
    response_model=list[EpochResponse],
    dependencies=[Depends(require_scopes("read"))],
)
def get_epochs(
    recording_id: uuid.UUID,
    from_ts: datetime = Query(..., alias="from"),
    to_ts: datetime = Query(..., alias="to"),
    tenant: TenantContext = Depends(get_tenant_context),
) -> list[EpochResponse]:
    _ensure_recording(recording_id, tenant)

    def _op(session):
        return (
            session.query(Epoch)
            .filter(Epoch.recording_id == recording_id)
            .filter(Epoch.tenant_id == tenant.id)
            .filter(Epoch.epoch_start_ts >= from_ts, Epoch.epoch_start_ts <= to_ts)
            .order_by(Epoch.epoch_index)
            .all()
        )

    rows = run_with_db_retry(_op, operation_name="recording_epochs")
    return [EpochResponse.model_validate(row) for row in rows]


@router.get(
    "/recordings/{recording_id}/predictions",
    response_model=list[PredictionResponse],
    dependencies=[Depends(require_scopes("read"))],
)
def get_predictions(
    recording_id: uuid.UUID,
    from_ts: datetime = Query(..., alias="from"),
    to_ts: datetime = Query(..., alias="to"),
    tenant: TenantContext = Depends(get_tenant_context),
) -> list[PredictionResponse]:
    _ensure_recording(recording_id, tenant)

    def _op(session):
        return (
            session.query(Prediction)
            .filter(Prediction.recording_id == recording_id)
            .filter(Prediction.tenant_id == tenant.id)
            .filter(
                Prediction.window_end_ts >= from_ts,
                Prediction.window_end_ts <= to_ts,
            )
            .order_by(Prediction.window_end_ts)
            .all()
        )

    rows = run_with_db_retry(_op, operation_name="recording_predictions")
    return [PredictionResponse.model_validate(row) for row in rows]


@router.get(
    "/recordings/{recording_id}/summary",
    response_model=RecordingSummary,
    dependencies=[Depends(require_scopes("read"))],
)
def get_summary(
    recording_id: uuid.UUID,
    from_ts: datetime = Query(..., alias="from"),
    to_ts: datetime = Query(..., alias="to"),
    tenant: TenantContext = Depends(get_tenant_context),
) -> RecordingSummary:
    _ensure_recording(recording_id, tenant)

    def _op(session):
        return (
            session.query(Prediction.predicted_stage)
            .filter(Prediction.recording_id == recording_id)
            .filter(Prediction.tenant_id == tenant.id)
            .filter(
                Prediction.window_end_ts >= from_ts,
                Prediction.window_end_ts <= to_ts,
            )
            .order_by(Prediction.window_end_ts)
            .all()
        )

    stages = run_with_db_retry(_op, operation_name="recording_summary")
    stage_list = [row[0] for row in stages]
    return summarize_predictions(str(recording_id), from_ts, to_ts, stage_list)


def _ensure_recording(recording_id: uuid.UUID, tenant: TenantContext) -> None:
    def _op(session):
        return (
            session.query(Recording)
            .filter(Recording.id == recording_id)
            .filter(Recording.tenant_id == tenant.id)
            .one_or_none()
        )

    recording = run_with_db_retry(_op, operation_name="recording_check")
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found"
        )
