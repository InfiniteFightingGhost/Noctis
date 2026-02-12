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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid stress mode"
        )
    window_size = (
        payload.window_size or model.metadata.get("window_size") or settings.window_size
    )
    result: dict[str, object] = {}

    if payload.mode in {"ingest", "both"}:
        ingest_result = await run_ingest_stress(
            model,
            feature_schema_version=settings.feature_schema_version,
            config=IngestConfig(
                device_count=payload.device_count,
                hours=payload.hours,
                epoch_seconds=payload.epoch_seconds,
                batch_size=payload.ingest_batch_size,
                seed=payload.seed,
                tenant_id=tenant.id,
            ),
        )
        result.update(
            {
                "device_count": ingest_result["device_count"],
                "recording_count": ingest_result["recording_count"],
                "epoch_count": ingest_result["epoch_count"],
                "ingest_duration_seconds": ingest_result["duration_seconds"],
                "ingest_throughput_per_sec": ingest_result["throughput_per_sec"],
            }
        )

    if payload.mode in {"inference", "both"}:
        inference_result = await anyio.to_thread.run_sync(
            run_inference_stress,
            model,
            iterations=payload.iterations,
            batch_size=payload.batch_size,
            window_size=int(window_size),
            seed=payload.seed,
        )
        result.update(
            {
                "iterations": payload.iterations,
                "batch_size": payload.batch_size,
                "window_size": int(window_size),
                "duration_seconds": inference_result["duration_seconds"],
                "avg_latency_ms": inference_result["avg_latency_ms"],
                "p95_latency_ms": inference_result["p95_latency_ms"],
                "throughput_per_sec": inference_result["throughput_per_sec"],
            }
        )
    memory_mb = memory_rss_mb()
    MEMORY_RSS_MB.set(memory_mb)
    STRESS_RUNS.inc()
    return StressRunResponse(
        mode=payload.mode,
        seed=payload.seed,
        iterations=result.get("iterations"),
        batch_size=result.get("batch_size"),
        window_size=result.get("window_size"),
        duration_seconds=result.get("duration_seconds"),
        avg_latency_ms=result.get("avg_latency_ms"),
        p95_latency_ms=result.get("p95_latency_ms"),
        throughput_per_sec=result.get("throughput_per_sec"),
        memory_rss_mb=memory_mb,
        device_count=result.get("device_count"),
        recording_count=result.get("recording_count"),
        epoch_count=result.get("epoch_count"),
        ingest_duration_seconds=result.get("ingest_duration_seconds"),
        ingest_throughput_per_sec=result.get("ingest_throughput_per_sec"),
    )
