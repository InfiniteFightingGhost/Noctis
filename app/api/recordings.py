from __future__ import annotations

from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.auth.dependencies import require_scopes
from app.db.models import Device, Epoch, Prediction, Recording
from app.db.session import run_with_db_retry
from app.schemas.epochs import EpochResponse
from app.schemas.predictions import PredictionResponse
from app.schemas.recordings import RecordingCreate, RecordingResponse, RecordingStartRequest
from app.schemas.summary import RecordingSummary
from app.services.device_ingest import close_open_recordings, resolve_device, resolve_recording
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")

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

    recording = run_with_db_retry(_create, commit=True, operation_name="recording_create")
    return RecordingResponse.model_validate(recording)


@router.post(
    "/recordings:start",
    response_model=RecordingResponse,
    dependencies=[Depends(require_scopes("ingest"))],
)
def start_recording(
    payload: RecordingStartRequest,
    tenant: TenantContext = Depends(get_tenant_context),
) -> RecordingResponse:
    def _op(session):
        device = resolve_device(
            session,
            tenant_id=tenant.id,
            device_id=None,
            external_id=payload.device_external_id,
            name=None,
        )
        # Rule 1: Close any existing open recordings for this device
        close_open_recordings(session, tenant_id=tenant.id, device_id=device.id)

        recording = resolve_recording(
            session,
            tenant_id=tenant.id,
            device_id=device.id,
            recording_id=None,
            started_at=payload.started_at,
            timezone_name=payload.timezone,
        )
        return recording

    recording = run_with_db_retry(_op, commit=True, operation_name="recording_start")
    return RecordingResponse.model_validate(recording)


@router.get(
    "/recordings",
    response_model=list[RecordingResponse],
    dependencies=[Depends(require_scopes("read"))],
)
def list_recordings(
    device_id: uuid.UUID | None = None,
    from_ts: datetime | None = Query(default=None, alias="from"),
    to_ts: datetime | None = Query(default=None, alias="to"),
    tenant: TenantContext = Depends(get_tenant_context),
) -> list[RecordingResponse]:
    def _op(session):
        query = session.query(Recording).filter(Recording.tenant_id == tenant.id)
        if device_id:
            query = query.filter(Recording.device_id == device_id)
        if from_ts:
            query = query.filter(Recording.started_at >= from_ts)
        if to_ts:
            query = query.filter(Recording.started_at <= to_ts)
        return query.order_by(Recording.started_at.desc()).all()

    recordings = run_with_db_retry(_op, operation_name="recordings_list")
    return [RecordingResponse.model_validate(recording) for recording in recordings]


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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")
