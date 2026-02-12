from __future__ import annotations

from datetime import datetime, timezone

from app.audit.service import run_audit
from app.db.models import Tenant
from app.db.session import run_with_db_retry


def run_audit_cycle() -> int:
    def _tenants(session):
        return session.query(Tenant).filter(Tenant.status == "active").all()

    tenants = run_with_db_retry(_tenants, operation_name="audit_tenants")
    total_issues = 0
    for tenant in tenants:

        def _audit(session):
            return run_audit(session, tenant_id=tenant.id)

        issues = run_with_db_retry(_audit, commit=True, operation_name="audit_run")
        total_issues += len(issues)
    return total_issues
