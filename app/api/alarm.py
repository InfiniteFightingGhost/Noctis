from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.auth.dependencies import require_scopes
from app.db.models import Alarm
from app.db.session import run_with_db_retry
from app.schemas.alarm import AlarmSettingsResponse, AlarmSettingsUpdate
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["alarm"])


@router.get(
    "/alarm",
    response_model=AlarmSettingsResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def list_alarms(
    tenant: TenantContext = Depends(get_tenant_context),
) -> AlarmSettingsResponse:
    alarm = run_with_db_retry(
        lambda session: _get_or_create_alarm(session, tenant.id),
        commit=True,
        operation_name="alarm_get_or_create",
    )
    return _to_settings_response(alarm)


@router.put(
    "/alarm",
    response_model=AlarmSettingsResponse,
    dependencies=[Depends(require_scopes("ingest"))],
)
def update_alarm(
    payload: AlarmSettingsUpdate,
    tenant: TenantContext = Depends(get_tenant_context),
) -> AlarmSettingsResponse:
    def _op(session):
        alarm = _get_or_create_alarm(session, tenant.id)
        if payload.wake_time is not None:
            alarm.wake_time = payload.wake_time
        if payload.wake_window_minutes is not None:
            alarm.wake_window_minutes = payload.wake_window_minutes
        if payload.sunrise_enabled is not None:
            alarm.sunrise_enabled = payload.sunrise_enabled
        if payload.sunrise_intensity is not None:
            alarm.sunrise_intensity = payload.sunrise_intensity
        if payload.sound_id is not None:
            alarm.sound_id = payload.sound_id
        alarm.updated_at = datetime.now(timezone.utc)
        session.add(alarm)
        session.flush()
        session.refresh(alarm)
        return alarm

    alarm = run_with_db_retry(_op, commit=True, operation_name="alarm_update")
    return _to_settings_response(alarm)


def _get_or_create_alarm(session, tenant_id) -> Alarm:
    alarm = (
        session.query(Alarm)
        .filter(Alarm.tenant_id == tenant_id)
        .order_by(Alarm.created_at.desc())
        .first()
    )
    if alarm:
        return alarm
    now = datetime.now(timezone.utc)
    alarm = Alarm(
        tenant_id=tenant_id,
        name="Default Alarm",
        scheduled_for=now,
        enabled=True,
        wake_time="06:45",
        wake_window_minutes=20,
        sunrise_enabled=True,
        sunrise_intensity=3,
        sound_id="ocean",
        created_at=now,
        updated_at=now,
    )
    session.add(alarm)
    session.flush()
    session.refresh(alarm)
    return alarm


def _to_settings_response(alarm: Alarm) -> AlarmSettingsResponse:
    return AlarmSettingsResponse(
        id=alarm.id,
        wake_time=alarm.wake_time,
        wake_window_minutes=alarm.wake_window_minutes,
        sunrise_enabled=alarm.sunrise_enabled,
        sunrise_intensity=alarm.sunrise_intensity,
        sound_id=alarm.sound_id,
        sound_options=[
            {"id": "ocean", "label": "Ocean Drift", "mood": "Calm"},
            {"id": "chimes", "label": "Sun Chimes", "mood": "Bright"},
            {"id": "forest", "label": "Forest Air", "mood": "Natural"},
        ],
        updated_at=alarm.updated_at,
    )
