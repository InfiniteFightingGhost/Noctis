from __future__ import annotations

import uuid

from app.governance.service import record_audit_log


class _Session:
    def __init__(self) -> None:
        self.added = []

    def add(self, item) -> None:
        self.added.append(item)


def test_record_audit_log_adds_entry() -> None:
    session = _Session()
    tenant_id = uuid.uuid4()
    record_audit_log(
        session,
        tenant_id=tenant_id,
        actor="tester",
        action="unit_test",
        target_type="model",
        target_id="v1",
        metadata={"source": "test"},
    )
    assert len(session.added) == 1
    entry = session.added[0]
    assert entry.tenant_id == tenant_id
    assert entry.action == "unit_test"
