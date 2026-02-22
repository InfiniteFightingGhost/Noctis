from __future__ import annotations

import anyio
from fastapi import APIRouter, Depends, Request

from app.core.metrics import MEMORY_RSS_MB, STRESS_RUNS
from app.core.settings import get_settings
from app.auth.dependencies import require_admin
from app.monitoring.memory import memory_rss_mb
from app.schemas.stress import StressRunRequest, StressRunResponse
from app.stress.service import IngestConfig, run_ingest_stress, run_inference_stress
from fastapi import HTTPException, status
from app.tenants.context import TenantContext, get_tenant_context
from app.db.session import run_with_db_retry
from app.feature_store.service import get_active_feature_schema


router = APIRouter(tags=["stress"], dependencies=[Depends(require_admin)])


@router.post("/stress/run", response_model=StressRunResponse)
async def stress_run(
    payload: StressRunRequest,
    request: Request,
    tenant: TenantContext = Depends(get_tenant_context),
) -> StressRunResponse:
    registry = request.app.state.model_registry
    model = registry.get_loaded()
    settings = get_settings()
    if payload.mode not in {"ingest", "inference", "both"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid stress mode")
    window_size = payload.window_size or model.metadata.get("window_size") or settings.window_size
    iterations: int | None = None
    batch_size: int | None = None
    window_size_value: int | None = None
    duration_seconds: float | None = None
    avg_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    throughput_per_sec: float | None = None
    device_count: int | None = None
    recording_count: int | None = None
    epoch_count: int | None = None
    ingest_duration_seconds: float | None = None
    ingest_throughput_per_sec: float | None = None

    if payload.mode in {"ingest", "both"}:
        schema = run_with_db_retry(
            lambda session: get_active_feature_schema(session),
            operation_name="stress_feature_schema",
        )
        if schema is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Active feature schema not found",
            )
        ingest_result = await run_ingest_stress(
            model,
            feature_schema_version=schema.version,
            config=IngestConfig(
                device_count=payload.device_count,
                hours=payload.hours,
                epoch_seconds=payload.epoch_seconds,
                batch_size=payload.ingest_batch_size,
                seed=payload.seed,
                tenant_id=tenant.id,
            ),
        )
        device_count = int(ingest_result["device_count"])
        recording_count = int(ingest_result["recording_count"])
        epoch_count = int(ingest_result["epoch_count"])
        ingest_duration_seconds = float(ingest_result["duration_seconds"])
        ingest_throughput_per_sec = float(ingest_result["throughput_per_sec"])

    if payload.mode in {"inference", "both"}:
        inference_result = await anyio.to_thread.run_sync(
            lambda: run_inference_stress(
                model,
                iterations=payload.iterations,
                batch_size=payload.batch_size,
                window_size=int(window_size),
                seed=payload.seed,
            )
        )
        iterations = payload.iterations
        batch_size = payload.batch_size
        window_size_value = int(window_size)
        duration_seconds = float(inference_result["duration_seconds"])
        avg_latency_ms = float(inference_result["avg_latency_ms"])
        p95_latency_ms = float(inference_result["p95_latency_ms"])
        throughput_per_sec = float(inference_result["throughput_per_sec"])
    memory_mb = memory_rss_mb()
    MEMORY_RSS_MB.set(memory_mb)
    STRESS_RUNS.inc()
    return StressRunResponse(
        mode=payload.mode,
        seed=payload.seed,
        iterations=iterations,
        batch_size=batch_size,
        window_size=window_size_value,
        duration_seconds=duration_seconds,
        avg_latency_ms=avg_latency_ms,
        p95_latency_ms=p95_latency_ms,
        throughput_per_sec=throughput_per_sec,
        memory_rss_mb=memory_mb,
        device_count=device_count,
        recording_count=recording_count,
        epoch_count=epoch_count,
        ingest_duration_seconds=ingest_duration_seconds,
        ingest_throughput_per_sec=ingest_throughput_per_sec,
    )
