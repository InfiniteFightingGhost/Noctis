from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.models import Device
from app.db.session import get_db
from app.schemas.devices import DeviceCreate, DeviceResponse


router = APIRouter(tags=["devices"])


@router.post("/devices", response_model=DeviceResponse)
def create_device(
    payload: DeviceCreate, db: Session = Depends(get_db)
) -> DeviceResponse:
    device = Device(name=payload.name, external_id=payload.external_id)
    db.add(device)
    db.commit()
    db.refresh(device)
    return DeviceResponse.model_validate(device)
