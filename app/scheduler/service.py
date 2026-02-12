from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import RetrainJob


def enqueue_retrain_job(
    session: Session,
    *,
    tenant_id,
    drift_score: float,
    triggering_features: list[dict[str, Any]],
    suggested_from_ts: datetime | None,
    suggested_to_ts: datetime | None,
    dataset_config: dict[str, Any],
    training_config: dict[str, Any],
) -> RetrainJob:
    job = RetrainJob(
        tenant_id=tenant_id,
        status="pending",
        drift_score=drift_score,
        triggering_features={"features": triggering_features},
        suggested_from_ts=suggested_from_ts,
        suggested_to_ts=suggested_to_ts,
        dataset_config=dataset_config,
        training_config=training_config,
    )
    session.add(job)
    session.flush()
    return job


def fetch_pending_jobs(session: Session, limit: int) -> list[RetrainJob]:
    return (
        session.execute(
            select(RetrainJob)
            .where(RetrainJob.status == "pending")
            .order_by(RetrainJob.created_at)
            .limit(limit)
        )
        .scalars()
        .all()
    )


def mark_job_running(session: Session, job: RetrainJob) -> None:
    job.status = "running"
    job.started_at = datetime.now(timezone.utc)
    session.add(job)


def mark_job_completed(
    session: Session,
    job: RetrainJob,
    *,
    model_version: str | None,
    error_message: str | None,
) -> None:
    job.status = "completed" if error_message is None else "failed"
    job.completed_at = datetime.now(timezone.utc)
    job.model_version = model_version
    job.error_message = error_message
    session.add(job)
