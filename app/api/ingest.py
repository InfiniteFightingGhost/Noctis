from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import require_scopes
from app.core.metrics import DEVICE_INGEST_RATE, INGEST_FAILURES, INGEST_REQUESTS
from app.core.settings import get_settings
from app.db.models import Recording
from app.db.session import run_with_db_retry
from app.schemas.epochs import EpochIngestBatch
from app.services.ingest import ingest_epochs
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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found"
            )

        settings = get_settings()
        rows = []
        for epoch in payload.epochs:
            if epoch.feature_schema_version != settings.feature_schema_version:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unsupported feature schema version",
                )
            rows.append(
                {
                    "tenant_id": tenant.id,
                    "recording_id": payload.recording_id,
                    "epoch_index": epoch.epoch_index,
                    "epoch_start_ts": epoch.epoch_start_ts,
                    "feature_schema_version": epoch.feature_schema_version,
                    "features_payload": {"features": epoch.features},
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
