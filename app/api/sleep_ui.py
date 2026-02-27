from __future__ import annotations

from datetime import timezone

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.context import AuthContext
from app.auth.dependencies import get_auth_context, require_scopes
from app.db.models import Device, Prediction, Recording
from app.db.session import run_with_db_retry
from app.schemas.sleep_ui import (
    HomeOverviewResponse,
    InsightFeedbackRequest,
    SleepSummaryResponse,
    SyncStatusResponse,
)
from app.services.summary import summarize_predictions
from app.services.user_identity import get_domain_user_for_auth
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["sleep-ui"])


@router.get(
    "/sleep/latest/summary",
    response_model=SleepSummaryResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def get_latest_sleep_summary(
    tenant: TenantContext = Depends(get_tenant_context),
    auth: AuthContext = Depends(get_auth_context),
) -> SleepSummaryResponse:
    user_id = None

    if auth.principal_type == "user":
        user_id = run_with_db_retry(
            lambda session: _get_user_id_for_auth(session, tenant_id=tenant.id, auth=auth),
            operation_name="sleep_ui_user_scope",
        )

    def _recording(session):
        query = session.query(Recording).filter(Recording.tenant_id == tenant.id)
        if user_id is not None:
            query = query.join(Device, Recording.device_id == Device.id).filter(
                Device.user_id == user_id
            )
        return query.order_by(Recording.started_at.desc()).first()

    recording = run_with_db_retry(_recording, operation_name="sleep_ui_latest_recording")
    if recording is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No recordings found")

    def _predictions(session):
        return (
            session.query(Prediction)
            .filter(Prediction.tenant_id == tenant.id)
            .filter(Prediction.recording_id == recording.id)
            .order_by(Prediction.window_end_ts)
            .all()
        )

    predictions = run_with_db_retry(_predictions, operation_name="sleep_ui_predictions")
    if not predictions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No predictions found")

    first = predictions[0].window_start_ts.astimezone(timezone.utc)
    last = predictions[-1].window_end_ts.astimezone(timezone.utc)
    summary = summarize_predictions(
        str(recording.id),
        first,
        last,
        [row.predicted_stage for row in predictions],
    )
    total_minutes = int(summary.total_minutes)
    score = max(40, min(95, int(summary.total_minutes / 5)))
    score_label = "Good" if score >= 75 else "Fair"
    deep_minutes = int(summary.time_in_stage_minutes.get("N3", 0))
    rem_minutes = int(summary.time_in_stage_minutes.get("R", 0))
    awake_minutes = int(summary.time_in_stage_minutes.get("W", 0))
    light_minutes = int(
        summary.time_in_stage_minutes.get("N1", 0) + summary.time_in_stage_minutes.get("N2", 0)
    )
    time_in_bed = max(total_minutes, 1)
    pct = {
        "awake": int((awake_minutes / time_in_bed) * 100),
        "light": int((light_minutes / time_in_bed) * 100),
        "deep": int((deep_minutes / time_in_bed) * 100),
        "rem": int((rem_minutes / time_in_bed) * 100),
    }
    return SleepSummaryResponse(
        recordingId=str(recording.id),
        dateLocal=recording.started_at.date().isoformat(),
        bedtimeLocal=recording.started_at.isoformat(),
        waketimeLocal=last.isoformat(),
        score=score,
        scoreLabel=score_label,
        totals={
            "totalSleepMin": total_minutes,
            "timeInBedMin": total_minutes,
            "sleepEfficiencyPct": max(0, 100 - pct["awake"]),
        },
        stages={
            "bins": [
                {
                    "startMinFromBedtime": index,
                    "durationMin": 1,
                    "stage": _stage_to_ui(prediction.predicted_stage),
                }
                for index, prediction in enumerate(predictions)
            ],
            "pct": pct,
        },
        metrics={
            "deepPct": pct["deep"],
            "avgHrBpm": 58,
            "avgRrBrpm": 14,
            "movementPct": 12,
        },
        insight={
            "text": "Your deepest sleep came in the first half of the night.",
            "tag": "pattern",
            "confidence": 0.78,
        },
        primaryAction={"label": "Improve Tonight", "action": "open_improve"},
        dataQuality={
            "status": "ok",
            "issues": [],
            "lastSyncAtLocal": last.isoformat(),
        },
    )


@router.get(
    "/sync/status",
    response_model=SyncStatusResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def get_sync_status(
    tenant: TenantContext = Depends(get_tenant_context),
) -> SyncStatusResponse:
    _ = tenant
    return SyncStatusResponse(status="ok")


@router.get(
    "/report/latest",
    response_model=SleepSummaryResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def get_latest_report(
    tenant: TenantContext = Depends(get_tenant_context),
    auth: AuthContext = Depends(get_auth_context),
) -> SleepSummaryResponse:
    return get_latest_sleep_summary(tenant, auth)


@router.get(
    "/home/overview",
    response_model=HomeOverviewResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def get_home_overview(
    tenant: TenantContext = Depends(get_tenant_context),
) -> HomeOverviewResponse:
    _ = tenant
    return HomeOverviewResponse(
        headline="Ready for tonight",
        lede="Review last night and tune your alarm and routine.",
        updated_at=None,
    )


@router.post(
    "/insights/feedback",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_scopes("read"))],
)
def post_insight_feedback(
    payload: InsightFeedbackRequest,
    tenant: TenantContext = Depends(get_tenant_context),
) -> dict[str, str]:
    _ = (tenant, payload)
    return {"status": "accepted"}


def _stage_to_ui(stage: str) -> str:
    mapping = {
        "W": "awake",
        "N1": "light",
        "N2": "light",
        "N3": "deep",
        "R": "rem",
    }
    return mapping.get(stage, "light")


def _get_user_id_for_auth(session, *, tenant_id, auth: AuthContext):
    user = get_domain_user_for_auth(session, tenant_id=tenant_id, auth=auth)
    return user.id if user else None
