from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import require_api_key
from app.core.settings import get_settings
from app.db.models import Recording
from app.db.session import get_db
from app.schemas.epochs import EpochIngestBatch
from app.services.ingest import ingest_epochs


router = APIRouter(tags=["epochs"], dependencies=[Depends(require_api_key)])


@router.post("/epochs:ingest")
def ingest_epoch_batch(
    payload: EpochIngestBatch, db: Session = Depends(get_db)
) -> dict:
    recording = db.get(Recording, payload.recording_id)
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
                "recording_id": payload.recording_id,
                "epoch_index": epoch.epoch_index,
                "epoch_start_ts": epoch.epoch_start_ts,
                "feature_schema_version": epoch.feature_schema_version,
                "features_payload": {"features": epoch.features},
            }
        )

    inserted = ingest_epochs(db, rows)
    logging.getLogger("app").info(
        "epochs_ingested",
        extra={
            "recording_id": str(payload.recording_id),
            "received": len(rows),
            "inserted": inserted,
        },
    )
    return {"inserted": inserted, "received": len(rows)}
