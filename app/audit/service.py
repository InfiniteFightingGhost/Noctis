from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.settings import get_settings
from app.db.models import AuditorReport, Epoch, Prediction, Recording
from app.feature_store.service import get_active_feature_schema


@dataclass(frozen=True)
class AuditIssue:
    issue_type: str
    severity: str
    recording_id: uuid.UUID | None


def run_audit(session: Session, *, tenant_id: uuid.UUID) -> list[AuditIssue]:
    settings = get_settings()
    schema = get_active_feature_schema(session)
    issues: list[AuditIssue] = []
    issues.extend(
        _check_missing_epochs(session, tenant_id, settings.audit_max_report_rows)
    )
    issues.extend(
        _check_window_gaps(session, tenant_id, settings.audit_epoch_gap_seconds)
    )
    issues.extend(
        _check_orphan_predictions(session, tenant_id, settings.audit_max_report_rows)
    )
    issues.extend(
        _check_invalid_feature_schema(
            session,
            tenant_id,
            schema.version,
            settings.audit_max_report_rows,
        )
    )
    for issue in issues:
        session.add(
            AuditorReport(
                tenant_id=tenant_id,
                issue_type=issue.issue_type,
                severity=issue.severity,
                recording_id=issue.recording_id,
                detected_at=datetime.now(timezone.utc),
            )
        )
    return issues


def _check_missing_epochs(
    session: Session, tenant_id: uuid.UUID, limit: int
) -> list[AuditIssue]:
    rows = (
        session.query(
            Epoch.recording_id,
            func.min(Epoch.epoch_index),
            func.max(Epoch.epoch_index),
            func.count(Epoch.epoch_index),
        )
        .filter(Epoch.tenant_id == tenant_id)
        .group_by(Epoch.recording_id)
        .limit(limit)
        .all()
    )
    issues: list[AuditIssue] = []
    for recording_id, min_idx, max_idx, count in rows:
        if min_idx is None or max_idx is None:
            continue
        expected = int(max_idx - min_idx + 1)
        missing = expected - int(count or 0)
        if missing > 0:
            severity = "HIGH" if missing >= 10 else "MEDIUM"
            issues.append(
                AuditIssue(
                    issue_type="missing_epochs",
                    severity=severity,
                    recording_id=recording_id,
                )
            )
    return issues


def _check_window_gaps(
    session: Session, tenant_id: uuid.UUID, gap_seconds: int
) -> list[AuditIssue]:
    rows = (
        session.query(
            Prediction.recording_id,
            Prediction.window_end_ts,
        )
        .filter(Prediction.tenant_id == tenant_id)
        .order_by(Prediction.recording_id, Prediction.window_end_ts)
        .all()
    )
    issues: list[AuditIssue] = []
    last_by_recording: dict[uuid.UUID, datetime] = {}
    for recording_id, window_end_ts in rows:
        last_ts = last_by_recording.get(recording_id)
        if last_ts is not None:
            delta = (window_end_ts - last_ts).total_seconds()
            if delta > gap_seconds:
                severity = "HIGH" if delta >= gap_seconds * 4 else "MEDIUM"
                issues.append(
                    AuditIssue(
                        issue_type="window_gap",
                        severity=severity,
                        recording_id=recording_id,
                    )
                )
        last_by_recording[recording_id] = window_end_ts
    return issues


def _check_orphan_predictions(
    session: Session, tenant_id: uuid.UUID, limit: int
) -> list[AuditIssue]:
    rows = (
        session.query(Prediction.recording_id)
        .outerjoin(
            Recording,
            (Prediction.recording_id == Recording.id)
            & (Recording.tenant_id == tenant_id),
        )
        .filter(Prediction.tenant_id == tenant_id)
        .filter(Recording.id.is_(None))
        .limit(limit)
        .all()
    )
    return [
        AuditIssue(
            issue_type="orphan_prediction",
            severity="HIGH",
            recording_id=row[0],
        )
        for row in rows
    ]


def _check_invalid_feature_schema(
    session: Session,
    tenant_id: uuid.UUID,
    expected_version: str,
    limit: int,
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    epoch_rows = (
        session.query(Epoch.recording_id)
        .filter(Epoch.tenant_id == tenant_id)
        .filter(Epoch.feature_schema_version != expected_version)
        .limit(limit)
        .all()
    )
    issues.extend(
        AuditIssue(
            issue_type="invalid_epoch_schema",
            severity="MEDIUM",
            recording_id=row[0],
        )
        for row in epoch_rows
    )
    prediction_rows = (
        session.query(Prediction.recording_id)
        .filter(Prediction.tenant_id == tenant_id)
        .filter(Prediction.feature_schema_version != expected_version)
        .limit(limit)
        .all()
    )
    issues.extend(
        AuditIssue(
            issue_type="invalid_prediction_schema",
            severity="MEDIUM",
            recording_id=row[0],
        )
        for row in prediction_rows
    )
    return issues
