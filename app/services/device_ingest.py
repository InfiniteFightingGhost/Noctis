from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import Device, DeviceEpochRaw, Recording
from app.feature_store.schema import FeatureSchemaRecord
from app.feature_store.validation import validate_feature_payload


METRIC_ALIASES: dict[str, str] = {
    "delta": "delta_power",
    "theta": "theta_power",
    "alpha": "alpha_power",
    "beta": "beta_power",
    "sigma": "sigma_power",
    "emg": "emg_rms",
    "eog": "eog_variance",
    "hr": "hr_mean",
    "hrv": "hrv_rmssd",
    "motion": "motion_index",
}


def resolve_device(
    session: Session,
    *,
    tenant_id: Any,
    device_id: Any | None,
    external_id: str | None,
    name: str | None,
) -> Device:
    if device_id is not None:
        device = (
            session.query(Device)
            .filter(Device.id == device_id)
            .filter(Device.tenant_id == tenant_id)
            .one_or_none()
        )
        if not device:
            raise ValueError("Device not found")
        return device
    if not external_id:
        raise ValueError("device_id or device_external_id is required")
    device = (
        session.query(Device)
        .filter(Device.tenant_id == tenant_id)
        .filter(Device.external_id == external_id)
        .one_or_none()
    )
    if device:
        return device
    device = Device(tenant_id=tenant_id, name=name or external_id, external_id=external_id)
    session.add(device)
    session.flush()
    session.refresh(device)
    return device


def resolve_recording(
    session: Session,
    *,
    tenant_id: Any,
    device_id: Any,
    recording_id: Any | None,
    started_at: datetime | None,
    timezone_name: str | None,
) -> Recording:
    if recording_id is not None:
        recording = (
            session.query(Recording)
            .filter(Recording.id == recording_id)
            .filter(Recording.tenant_id == tenant_id)
            .filter(Recording.device_id == device_id)
            .one_or_none()
        )
        if not recording:
            raise ValueError("Recording not found")
        return recording
    recording = (
        session.query(Recording)
        .filter(Recording.tenant_id == tenant_id)
        .filter(Recording.device_id == device_id)
        .filter(Recording.ended_at.is_(None))
        .order_by(desc(Recording.started_at))
        .first()
    )
    if recording:
        return recording
    started_at = started_at or datetime.now(timezone.utc)
    recording = Recording(
        tenant_id=tenant_id,
        device_id=device_id,
        started_at=started_at,
        timezone=timezone_name,
    )
    session.add(recording)
    session.flush()
    session.refresh(recording)
    return recording


def normalize_metrics(metrics: Any, schema: FeatureSchemaRecord) -> Any:
    if isinstance(metrics, dict):
        normalized: dict[str, Any] = {}
        for key, value in metrics.items():
            normalized_key = METRIC_ALIASES.get(str(key), str(key))
            normalized[normalized_key] = value
        return normalized
    return metrics


def build_epoch_rows(
    *,
    tenant_id: Any,
    device_id: Any,
    recording_id: Any,
    epochs: list[Any],
    schema: FeatureSchemaRecord,
) -> tuple[list[dict], list[dict]]:
    raw_rows: list[dict] = []
    epoch_rows: list[dict] = []
    for epoch in epochs:
        normalized = normalize_metrics(epoch.metrics, schema)
        validated = validate_feature_payload(normalized, schema)
        raw_rows.append(
            {
                "tenant_id": tenant_id,
                "device_id": device_id,
                "recording_id": recording_id,
                "epoch_index": epoch.epoch_index,
                "epoch_start_ts": epoch.epoch_start_ts,
                "raw_metrics": epoch.metrics,
            }
        )
        epoch_rows.append(
            {
                "tenant_id": tenant_id,
                "recording_id": recording_id,
                "epoch_index": epoch.epoch_index,
                "epoch_start_ts": epoch.epoch_start_ts,
                "feature_schema_version": schema.version,
                "features_payload": {"features": validated},
            }
        )
    return raw_rows, epoch_rows


def store_device_epoch_raw(session: Session, rows: list[dict]) -> int:
    if not rows:
        return 0
    stmt = insert(DeviceEpochRaw).values(rows)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=[
            DeviceEpochRaw.tenant_id,
            DeviceEpochRaw.recording_id,
            DeviceEpochRaw.epoch_start_ts,
        ]
    )
    result = session.execute(stmt.returning(DeviceEpochRaw.id))
    return len(result.fetchall())
