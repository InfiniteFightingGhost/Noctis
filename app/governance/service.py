from __future__ import annotations

import uuid
from typing import Any

from app.db.models import AuditLog
from app.db.session import run_with_db_retry


def record_audit_log(
    session,
    *,
    tenant_id: uuid.UUID,
    actor: str,
    action: str,
    target_type: str,
    target_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    session.add(
        AuditLog(
            tenant_id=tenant_id,
            actor=actor,
            action=action,
            target_type=target_type,
            target_id=target_id,
            metadata_json=metadata,
        )
    )


def record_audit_log_with_retry(
    *,
    tenant_id: uuid.UUID,
    actor: str,
    action: str,
    target_type: str,
    target_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    def _op(session):
        record_audit_log(
            session,
            tenant_id=tenant_id,
            actor=actor,
            action=action,
            target_type=target_type,
            target_id=target_id,
            metadata=metadata,
        )

    run_with_db_retry(_op, commit=True, operation_name="audit_log")
