from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.models import ModelUsageStat, Prediction
from app.db.session import get_circuit_breaker, get_engine
from app.monitoring.memory import memory_rss_mb


def build_performance_snapshot(
    session: Session, *, tenant_id, started_at: datetime, sample_size: int
) -> dict[str, object]:
    pool = get_engine().pool
    breaker = get_circuit_breaker()
    now = datetime.now(timezone.utc)

    usage_rows = (
        session.query(ModelUsageStat)
        .filter(ModelUsageStat.tenant_id == tenant_id)
        .order_by(ModelUsageStat.created_at.desc())
        .limit(sample_size)
        .all()
    )
    inference_metrics = _compute_inference_metrics(usage_rows)
    db_write_speed = _compute_db_write_speed(session, tenant_id, sample_size)
    slow_queries = _fetch_slow_queries(session)

    return {
        "timestamp": now,
        "memory_rss_mb": memory_rss_mb(),
        "db_pool": {
            "size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "status": pool.status(),
        },
        "uptime_seconds": (now - started_at).total_seconds(),
        "circuit_breaker_state": breaker.state,
        "inference_timing": inference_metrics,
        "db_write_speed": db_write_speed,
        "slow_queries": slow_queries,
    }


def _compute_inference_metrics(rows: list[ModelUsageStat]) -> dict[str, object]:
    if not rows:
        return {
            "sample_size": 0,
            "prediction_count": 0,
            "average_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
        }
    prediction_count = sum(row.prediction_count for row in rows)
    weighted_latency = sum(
        row.prediction_count * row.average_latency_ms for row in rows
    )
    avg_latency = weighted_latency / prediction_count if prediction_count else 0.0
    latencies = sorted(row.average_latency_ms for row in rows)
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    p99 = latencies[int(0.99 * (len(latencies) - 1))]
    return {
        "sample_size": len(rows),
        "prediction_count": prediction_count,
        "average_latency_ms": avg_latency,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
    }


def _compute_db_write_speed(
    session: Session, tenant_id, sample_size: int
) -> dict[str, object]:
    rows = (
        session.query(Prediction.created_at)
        .filter(Prediction.tenant_id == tenant_id)
        .order_by(Prediction.created_at.desc())
        .limit(sample_size)
        .all()
    )
    if len(rows) < 2:
        return {"sample_size": len(rows), "window_seconds": 0.0, "writes_per_sec": 0.0}
    timestamps = [row[0] for row in rows]
    newest = max(timestamps)
    oldest = min(timestamps)
    window_seconds = (newest - oldest).total_seconds()
    writes_per_sec = (len(rows) / window_seconds) if window_seconds > 0 else 0.0
    return {
        "sample_size": len(rows),
        "window_seconds": window_seconds,
        "writes_per_sec": writes_per_sec,
    }


def _fetch_slow_queries(session: Session) -> list[dict[str, object]]:
    try:
        rows = session.execute(
            text(
                "SELECT query, total_time, calls "
                "FROM pg_stat_statements "
                "ORDER BY total_time DESC "
                "LIMIT 5"
            )
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [
        {"query": row[0], "total_time_ms": float(row[1]), "calls": int(row[2])}
        for row in rows
    ]
