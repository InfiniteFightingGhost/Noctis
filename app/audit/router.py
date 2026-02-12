from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.auth.dependencies import require_admin
from app.core.settings import get_settings
from app.db.models import AuditorReport
from app.db.session import run_with_db_retry
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["audit"], dependencies=[Depends(require_admin)])


@router.get("/audit/report")
def audit_report(
    tenant: TenantContext = Depends(get_tenant_context),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict:
    settings = get_settings()
    effective_limit = min(limit, settings.audit_max_report_rows)

    def _op(session):
        rows = (
            session.query(AuditorReport)
            .filter(AuditorReport.tenant_id == tenant.id)
            .order_by(AuditorReport.detected_at.desc())
            .limit(effective_limit)
            .all()
        )
        return [
            {
                "id": str(row.id),
                "tenant_id": str(row.tenant_id),
                "issue_type": row.issue_type,
                "severity": row.severity,
                "recording_id": str(row.recording_id) if row.recording_id else None,
                "detected_at": row.detected_at,
            }
            for row in rows
        ]

    payload = run_with_db_retry(_op, operation_name="audit_report")
    return {"reports": payload}
