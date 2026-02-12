from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.ml.registry import ModelRegistry
from app.monitoring.memory import memory_rss_mb


def build_monitoring_summary(db: Session, registry: ModelRegistry) -> dict[str, object]:
    db.execute(text("SELECT 1"))
    loaded = registry.get_loaded()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": {"status": "ok"},
        "model": {"status": "ok", "version": loaded.version},
        "memory": {"rss_mb": memory_rss_mb()},
    }
