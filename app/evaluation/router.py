from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import require_scopes
from app.core.metrics import EVALUATION_REQUESTS
from app.db.session import run_with_db_retry
from app.core.settings import get_settings
from app.evaluation.service import (
    compute_confidence_drift,
    compute_evaluation,
    compute_model_usage_stats,
    compute_rolling_evaluation,
)
from app.schemas.evaluation import EvaluationResponse
from app.db.models import EvaluationMetric, Recording
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["evaluation"], dependencies=[Depends(require_scopes("read"))])


@router.get("/recordings/{recording_id}/evaluation", response_model=EvaluationResponse)
def recording_evaluation(
    recording_id: uuid.UUID,
    from_ts: datetime | None = Query(default=None, alias="from"),
    to_ts: datetime | None = Query(default=None, alias="to"),
    tenant: TenantContext = Depends(get_tenant_context),
) -> dict:
    _ensure_recording(recording_id, tenant)

    def _op(session):
        return compute_evaluation(
            session,
            tenant_id=tenant.id,
            recording_id=recording_id,
            from_ts=from_ts,
            to_ts=to_ts,
        )

    EVALUATION_REQUESTS.labels(scope="recording").inc()
    payload = run_with_db_retry(_op, operation_name="recording_evaluation")
    payload.update(
        {
            "scope": "recording",
            "recording_id": recording_id,
            "model_version": None,
            "from_ts": from_ts,
            "to_ts": to_ts,
        }
    )
    _store_evaluation(
        payload,
        tenant_id=tenant.id,
        recording_id=recording_id,
        model_version=None,
    )
    return payload


@router.get("/model/evaluation/global", response_model=EvaluationResponse)
def global_evaluation(
    model_version: str | None = None,
    from_ts: datetime | None = Query(default=None, alias="from"),
    to_ts: datetime | None = Query(default=None, alias="to"),
    tenant: TenantContext = Depends(get_tenant_context),
) -> dict:
    settings = get_settings()

    def _op(session):
        evaluation = compute_evaluation(
            session,
            tenant_id=tenant.id,
            model_version=model_version,
            from_ts=from_ts,
            to_ts=to_ts,
        )
        rolling_full = compute_rolling_evaluation(
            session, tenant_id=tenant.id, model_version=model_version, days=7
        )
        rolling = _rolling_summary(rolling_full)
        confidence_drift = compute_confidence_drift(
            session, tenant_id=tenant.id, model_version=model_version, days=7
        )
        model_usage = compute_model_usage_stats(
            session,
            tenant_id=tenant.id,
            model_version=model_version,
            from_ts=from_ts,
            to_ts=to_ts,
        )
        evaluation.update(
            {
                "rolling_7_day": rolling,
                "confidence_drift": confidence_drift,
                "model_usage_stats": model_usage,
                "confidence_drift_threshold": settings.drift_z_threshold,
            }
        )
        return evaluation

    EVALUATION_REQUESTS.labels(scope="global").inc()
    payload = run_with_db_retry(_op, operation_name="global_evaluation")
    payload.update(
        {
            "scope": "global",
            "recording_id": None,
            "model_version": model_version,
            "from_ts": from_ts,
            "to_ts": to_ts,
        }
    )
    _store_evaluation(
        payload,
        tenant_id=tenant.id,
        recording_id=None,
        model_version=model_version,
    )
    return payload


def _rolling_summary(payload: dict) -> dict:
    return {
        "total_predictions": payload.get("total_predictions", 0),
        "labeled_predictions": payload.get("labeled_predictions", 0),
        "accuracy": payload.get("accuracy"),
        "macro_f1": payload.get("macro_f1"),
        "average_confidence": payload.get("average_confidence", 0.0),
        "prediction_distribution": payload.get("prediction_distribution", {}),
        "per_class_frequency": payload.get("per_class_frequency", []),
        "entropy": payload.get("entropy", {}),
    }


def _store_evaluation(
    payload: dict,
    tenant_id: uuid.UUID,
    recording_id: uuid.UUID | None,
    model_version: str | None,
) -> None:
    def _op(session):
        session.add(
            EvaluationMetric(
                tenant_id=tenant_id,
                model_version=model_version or "unknown",
                recording_id=recording_id,
                from_ts=payload.get("from_ts"),
                to_ts=payload.get("to_ts"),
                metrics={
                    k: _normalize_metric_value(v)
                    for k, v in payload.items()
                    if k not in {"from_ts", "to_ts"}
                },
            )
        )

    run_with_db_retry(_op, commit=True, operation_name="evaluation_store")


def _normalize_metric_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, dict):
        return {key: _normalize_metric_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_metric_value(item) for item in value]
    return value


def _ensure_recording(recording_id: uuid.UUID, tenant: TenantContext) -> None:
    def _op(session):
        return (
            session.query(Recording)
            .filter(Recording.id == recording_id)
            .filter(Recording.tenant_id == tenant.id)
            .one_or_none()
        )

    recording = run_with_db_retry(_op, operation_name="evaluation_recording")
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
