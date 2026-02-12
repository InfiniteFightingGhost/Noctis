from __future__ import annotations

from dataclasses import dataclass

from app.timescale_ops import service as timescale_service


@dataclass(frozen=True)
class _Settings:
    epochs_retention_days: int = 30
    predictions_retention_days: int = 60
    epochs_compression_after_days: int = 3
    timescale_compression_segmentby: str = "recording_id"
    timescale_chunk_interval_days: int = 1


def test_build_policy_actions_uses_settings(monkeypatch) -> None:
    monkeypatch.setattr(timescale_service, "get_settings", lambda: _Settings())
    actions = timescale_service.build_policy_actions()
    sql = " ".join(action.sql for action in actions)
    assert "30 days" in sql
    assert "60 days" in sql
    assert "3 days" in sql
    assert "set_chunk_time_interval" in sql
