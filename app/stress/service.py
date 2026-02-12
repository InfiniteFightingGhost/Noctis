from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
import uuid

import anyio
import numpy as np

from app.db.models import Device, Recording
from app.db.session import run_with_db_retry
from app.ml.registry import LoadedModel
from app.services.ingest import ingest_epochs
from app.services.inference import predict_windows
from app.stress.simulator import SyntheticFeatureGenerator


@dataclass(frozen=True)
class IngestConfig:
    device_count: int
    hours: int
    epoch_seconds: int
    batch_size: int
    seed: int
    tenant_id: uuid.UUID


def run_inference_stress(
    model: LoadedModel,
    *,
    iterations: int,
    batch_size: int,
    window_size: int,
    seed: int,
) -> dict[str, Any]:
    if iterations <= 0 or batch_size <= 0 or window_size <= 0:
        raise ValueError("iterations, batch_size, window_size must be positive")
    rng = np.random.default_rng(seed)
    latencies: list[float] = []
    start = time.perf_counter()
    for _ in range(iterations):
        batch = rng.random(
            (batch_size, window_size, model.feature_schema.size), dtype=np.float32
        )
        t0 = time.perf_counter()
        predict_windows(model, [window for window in batch])
        latencies.append((time.perf_counter() - t0) * 1000.0)
    duration = time.perf_counter() - start
    latencies_sorted = sorted(latencies)
    p95 = latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))]
    avg = sum(latencies_sorted) / len(latencies_sorted)
    throughput = (iterations * batch_size) / duration if duration > 0 else 0.0
    return {
        "duration_seconds": duration,
        "avg_latency_ms": avg,
        "p95_latency_ms": p95,
        "throughput_per_sec": throughput,
    }


async def run_ingest_stress(
    model: LoadedModel,
    *,
    feature_schema_version: str,
    config: IngestConfig,
) -> dict[str, Any]:
    if config.device_count <= 0 or config.hours <= 0 or config.epoch_seconds <= 0:
        raise ValueError("device_count, hours, epoch_seconds must be positive")
    total_epochs = int((config.hours * 3600) / config.epoch_seconds)
    start_ts = datetime.now(timezone.utc) - timedelta(hours=config.hours)
    generator = SyntheticFeatureGenerator(
        feature_size=model.feature_schema.size,
        seed=config.seed,
    )

    async def _run_device(device_index: int) -> int:
        device_name = f"stress-device-{device_index}"
        recording_id = await anyio.to_thread.run_sync(
            _create_device_recording,
            device_name,
            start_ts,
            config.tenant_id,
        )
        inserted = 0
        for batch_start in range(0, total_epochs, config.batch_size):
            batch = generator.generate_epoch_batch(
                device_index=device_index,
                start_ts=start_ts,
                epoch_seconds=config.epoch_seconds,
                start_index=batch_start,
                count=min(config.batch_size, total_epochs - batch_start),
                total_epochs=total_epochs,
                feature_schema_version=feature_schema_version,
            )
            inserted += await anyio.to_thread.run_sync(
                _ingest_batch,
                recording_id,
                config.tenant_id,
                batch,
            )
        return inserted

    start = time.perf_counter()
    inserted_total = 0
    results: list[int] = []
    async with anyio.create_task_group() as group:

        async def _task(device_index: int) -> None:
            results.append(await _run_device(device_index))

        for idx in range(config.device_count):
            group.start_soon(_task, idx)

    inserted_total = sum(results)
    duration = time.perf_counter() - start
    throughput = inserted_total / duration if duration > 0 else 0.0
    return {
        "device_count": config.device_count,
        "recording_count": config.device_count,
        "epoch_count": inserted_total,
        "duration_seconds": duration,
        "throughput_per_sec": throughput,
    }


def _create_device_recording(
    device_name: str, start_ts: datetime, tenant_id: uuid.UUID
):
    def _op(session):
        device = Device(name=device_name, tenant_id=tenant_id)
        session.add(device)
        session.flush()
        recording = Recording(
            tenant_id=tenant_id,
            device_id=device.id,
            started_at=start_ts,
            timezone="UTC",
        )
        session.add(recording)
        session.flush()
        return recording.id

    return run_with_db_retry(_op, commit=True, operation_name="stress_device")


def _ingest_batch(
    recording_id, tenant_id: uuid.UUID, batch: list[dict[str, Any]]
) -> int:
    rows = [
        {
            "tenant_id": tenant_id,
            "recording_id": recording_id,
            "epoch_index": epoch["epoch_index"],
            "epoch_start_ts": epoch["epoch_start_ts"],
            "feature_schema_version": epoch["feature_schema_version"],
            "features_payload": {"features": epoch["features"]},
        }
        for epoch in batch
    ]
    return run_with_db_retry(
        lambda session: ingest_epochs(session, rows),
        commit=True,
        operation_name="stress_ingest",
    )
