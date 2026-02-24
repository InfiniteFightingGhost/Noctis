from __future__ import annotations

from datetime import datetime, timedelta, timezone
import uuid

from app.api import sleep_ui
from app.auth.context import AuthContext
from app.tenants.context import TenantContext


class _Recording:
    def __init__(self) -> None:
        self.id = uuid.uuid4()
        self.started_at = datetime.now(timezone.utc) - timedelta(hours=7)


class _Prediction:
    def __init__(self, index: int) -> None:
        now = datetime.now(timezone.utc)
        self.window_start_ts = now + timedelta(minutes=index)
        self.window_end_ts = now + timedelta(minutes=index + 1)
        self.predicted_stage = "N2"


def _tenant() -> TenantContext:
    return TenantContext(id=uuid.uuid4(), name="tenant", status="active")


def _auth(principal_type: str) -> AuthContext:
    return AuthContext(
        client_id=uuid.uuid4(),
        client_name="user@example.com",
        role="user" if principal_type == "user" else "read",
        tenant_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        scopes={"read", "ingest"},
        key_id="kid",
        principal_type=principal_type,
    )


def test_sleep_summary_user_principal_uses_user_scope(monkeypatch) -> None:
    called_ops: list[str] = []

    def fake_run_with_db_retry(_op, *, operation_name: str, **_kwargs):
        called_ops.append(operation_name)
        if operation_name == "sleep_ui_user_scope":
            return uuid.uuid4()
        if operation_name == "sleep_ui_latest_recording":
            return _Recording()
        if operation_name == "sleep_ui_predictions":
            return [_Prediction(index) for index in range(5)]
        raise AssertionError("Unexpected operation")

    monkeypatch.setattr(sleep_ui, "run_with_db_retry", fake_run_with_db_retry)

    response = sleep_ui.get_latest_sleep_summary(_tenant(), _auth("user"))

    assert response.recordingId
    assert "sleep_ui_user_scope" in called_ops


def test_sleep_summary_service_principal_skips_user_scope(monkeypatch) -> None:
    called_ops: list[str] = []

    def fake_run_with_db_retry(_op, *, operation_name: str, **_kwargs):
        called_ops.append(operation_name)
        if operation_name == "sleep_ui_latest_recording":
            return _Recording()
        if operation_name == "sleep_ui_predictions":
            return [_Prediction(index) for index in range(5)]
        raise AssertionError("Unexpected operation")

    monkeypatch.setattr(sleep_ui, "run_with_db_retry", fake_run_with_db_retry)

    response = sleep_ui.get_latest_sleep_summary(_tenant(), _auth("service"))

    assert response.recordingId
    assert "sleep_ui_user_scope" not in called_ops
