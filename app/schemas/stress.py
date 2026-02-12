from __future__ import annotations

from app.schemas.common import BaseSchema


class StressRunRequest(BaseSchema):
    mode: str = "ingest"
    device_count: int = 10
    hours: int = 8
    epoch_seconds: int = 30
    ingest_batch_size: int = 256
    iterations: int = 200
    batch_size: int = 32
    seed: int = 42
    window_size: int | None = None


class StressRunResponse(BaseSchema):
    mode: str
    seed: int
    iterations: int | None = None
    batch_size: int | None = None
    window_size: int | None = None
    duration_seconds: float | None = None
    avg_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    throughput_per_sec: float | None = None
    memory_rss_mb: float | None = None
    device_count: int | None = None
    recording_count: int | None = None
    epoch_count: int | None = None
    ingest_duration_seconds: float | None = None
    ingest_throughput_per_sec: float | None = None
