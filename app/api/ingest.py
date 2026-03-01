from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from app.auth.context import AuthContext
from app.auth.dependencies import get_auth_context, require_scopes
from app.core.metrics import DEVICE_INGEST_RATE, INGEST_FAILURES, INGEST_REQUESTS
from app.db.models import Device, Recording
from app.db.session import run_with_db_retry
from app.feature_store.service import get_active_feature_schema
from app.feature_store.schema import FeatureSchemaRecord
from app.feature_store.validation import validate_feature_payload
from app.schemas.device_ingest import DeviceEpochIngestBatch
from app.schemas.epochs import EpochIngestBatch
from app.services.device_ingest import (
    build_epoch_rows,
    resolve_device,
    resolve_recording,
    store_device_epoch_raw,
)
from app.services.ingest import ingest_epochs
from app.services.auto_predict import auto_predict_recording
from app.services.user_identity import resolve_or_create_domain_user_for_auth
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["epochs"], dependencies=[Depends(require_scopes("ingest"))])


@router.post("/epochs:ingest")
def ingest_epoch_batch(
    payload: EpochIngestBatch,
    tenant: TenantContext = Depends(get_tenant_context),
) -> dict:
    INGEST_REQUESTS.inc()

    def _recording(session):
        return (
            session.query(Recording)
            .filter(Recording.id == payload.recording_id)
            .filter(Recording.tenant_id == tenant.id)
            .one_or_none()
        )

    try:
        recording = run_with_db_retry(_recording, operation_name="ingest_recording")
        if not recording:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")

        schema = cast(
            FeatureSchemaRecord,
            run_with_db_retry(
                lambda session: get_active_feature_schema(session),
                operation_name="ingest_feature_schema",
            ),
        )
        rows = []
        for epoch in payload.epochs:
            if epoch.feature_schema_version != schema.version:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unsupported feature schema version",
                )
            validated_features = validate_feature_payload(epoch.features, schema)
            rows.append(
                {
                    "tenant_id": tenant.id,
                    "recording_id": payload.recording_id,
                    "epoch_index": epoch.epoch_index,
                    "epoch_start_ts": epoch.epoch_start_ts,
                    "feature_schema_version": epoch.feature_schema_version,
                    "features_payload": {"features": validated_features},
                }
            )

        start = time.perf_counter()
        inserted = run_with_db_retry(
            lambda session: ingest_epochs(session, rows),
            commit=True,
            operation_name="ingest_epochs",
        )
        duration = time.perf_counter() - start
        if duration > 0:
            DEVICE_INGEST_RATE.set(inserted / duration)
        logging.getLogger("app").info(
            "epochs_ingested",
            extra={
                "recording_id": str(payload.recording_id),
                "received": len(rows),
                "inserted": inserted,
                "tenant_id": str(tenant.id),
            },
        )
        return {"inserted": inserted, "received": len(rows)}
    except HTTPException:
        INGEST_FAILURES.inc()
        raise
    except Exception:
        INGEST_FAILURES.inc()
        raise


@router.post("/epochs:ingest-device")
def ingest_device_epoch_batch(
    payload: DeviceEpochIngestBatch,
    tenant: TenantContext = Depends(get_tenant_context),
    auth: AuthContext = Depends(get_auth_context),
    background_tasks: BackgroundTasks = None,
    request: Request | None = None,
) -> dict:
    INGEST_REQUESTS.inc()

    schema = run_with_db_retry(
        lambda session: get_active_feature_schema(session),
        operation_name="ingest_feature_schema",
    )
    if schema is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Active feature schema unavailable",
        )
    started_at = payload.recording_started_at
    if started_at is None:
        started_at = min(
            (epoch.epoch_start_ts for epoch in payload.epochs),
            default=None,
        )
    if started_at is None:
        started_at = datetime.now(timezone.utc)

    for epoch in payload.epochs:
        if epoch.feature_schema_version and epoch.feature_schema_version != schema.version:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported feature schema version",
            )

    def _op(session):
        if auth.principal_type == "user":
            user = resolve_or_create_domain_user_for_auth(session, tenant_id=tenant.id, auth=auth)
            device = _resolve_existing_device(session, tenant_id=tenant.id, payload=payload)
            if device is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Device not found"
                )
            _ensure_user_device_binding(device_user_id=device.user_id, current_user_id=user.id)
        else:
            device = _resolve_service_device(session, tenant_id=tenant.id, payload=payload)
        recording = resolve_recording(
            session,
            tenant_id=tenant.id,
            device_id=device.id,
            recording_id=payload.recording_id,
            started_at=started_at,
            timezone_name=payload.timezone,
        )
        raw_rows, epoch_rows = build_epoch_rows(
            tenant_id=tenant.id,
            device_id=device.id,
            recording_id=recording.id,
            epochs=payload.epochs,
            schema=schema,
        )
        raw_inserted = store_device_epoch_raw(session, raw_rows)
        inserted = 0
        if payload.forward_to_ml:
            inserted = ingest_epochs(session, epoch_rows)
        return device, recording, inserted, raw_inserted, len(epoch_rows)

    start = time.perf_counter()
    try:
        device, recording, inserted, raw_inserted, received = run_with_db_retry(
            _op,
            commit=True,
            operation_name="ingest_device_epochs",
        )
        duration = time.perf_counter() - start
        if duration > 0 and payload.forward_to_ml:
            DEVICE_INGEST_RATE.set(inserted / duration)
        if payload.forward_to_ml and request is not None and background_tasks is not None:
            background_tasks.add_task(
                auto_predict_recording,
                tenant_id=tenant.id,
                recording_id=recording.id,
                registry=request.app.state.model_registry,
            )

        logging.getLogger("app").info(
            "device_epochs_ingested",
            extra={
                "device_id": str(device.id),
                "recording_id": str(recording.id),
                "received": received,
                "inserted": inserted,
                "raw_inserted": raw_inserted,
                "tenant_id": str(tenant.id),
            },
        )
        return {
            "device_id": str(device.id),
            "recording_id": str(recording.id),
            "received": received,
            "inserted": inserted,
            "raw_inserted": raw_inserted,
            "forwarded": payload.forward_to_ml,
        }
    except HTTPException:
        INGEST_FAILURES.inc()
        raise
    except ValueError as exc:
        INGEST_FAILURES.inc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception:
        INGEST_FAILURES.inc()
        raise


def _resolve_existing_device(
    session, *, tenant_id, payload: DeviceEpochIngestBatch
) -> Device | None:
    query = session.query(Device).filter(Device.tenant_id == tenant_id)
    if payload.device_id is not None:
        return query.filter(Device.id == payload.device_id).one_or_none()
    return query.filter(Device.external_id == payload.device_external_id).one_or_none()


def _resolve_service_device(session, *, tenant_id, payload: DeviceEpochIngestBatch) -> Device:
    existing = _resolve_existing_device(session, tenant_id=tenant_id, payload=payload)
    if existing is not None:
        return existing
    if payload.device_id is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    if not payload.device_external_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Device identifier is required"
        )
    if not payload.device_name:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found; provide device_name to register device",
        )
    return resolve_device(
        session,
        tenant_id=tenant_id,
        device_id=None,
        external_id=payload.device_external_id,
        name=payload.device_name,
    )


def _ensure_user_device_binding(*, device_user_id, current_user_id) -> None:
    if device_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Device is not paired to a user",
        )
    if device_user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Device is paired to a different user",
        )
