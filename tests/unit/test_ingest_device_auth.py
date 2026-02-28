from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from app.api import ingest
from app.schemas.device_ingest import DeviceEpochIngest, DeviceEpochIngestBatch


def _payload(
    *, external_id: str = "ext-1", device_name: str | None = None
) -> DeviceEpochIngestBatch:
    return DeviceEpochIngestBatch(
        device_external_id=external_id,
        device_name=device_name,
        epochs=[
            DeviceEpochIngest(
                epoch_index=0,
                epoch_start_ts="2026-02-24T00:00:00Z",
                metrics=[0.1] * 10,
            )
        ],
    )


def test_ensure_user_device_binding_rejects_cross_user() -> None:
    with pytest.raises(HTTPException) as exc:
        ingest._ensure_user_device_binding(
            device_user_id=uuid.uuid4(), current_user_id=uuid.uuid4()
        )

    assert exc.value.status_code == 403
    assert "different user" in str(exc.value.detail)


def test_resolve_service_device_requires_explicit_name_for_auto_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest, "_resolve_existing_device", lambda *args, **kwargs: None)

    with pytest.raises(HTTPException) as exc:
        ingest._resolve_service_device(
            None, tenant_id=uuid.uuid4(), payload=_payload(device_name=None)
        )

    assert exc.value.status_code == 404
    assert "provide device_name" in str(exc.value.detail)
