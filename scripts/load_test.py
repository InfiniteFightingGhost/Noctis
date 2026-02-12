from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

from pathlib import Path

import httpx

from app.ml.feature_schema import load_feature_schema
from app.stress.simulator import SyntheticFeatureGenerator


async def main() -> None:
    parser = argparse.ArgumentParser(description="Noctis load test runner")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-token", default="changeme")
    parser.add_argument("--admin-token", default="adminchangeme")
    parser.add_argument("--device-count", type=int, default=25)
    parser.add_argument("--hours", type=int, default=8)
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--ingest-batch-size", type=int, default=256)
    parser.add_argument("--predict-concurrency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reload-during", action="store_true")
    args = parser.parse_args()

    total_epochs = int((args.hours * 3600) / args.epoch_seconds)
    start_ts = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    schema = load_feature_schema(Path("models/active/feature_schema.json"))
    generator = SyntheticFeatureGenerator(feature_size=schema.size, seed=args.seed)

    async with httpx.AsyncClient(base_url=args.base_url, timeout=60.0) as client:
        device_ids = []
        recording_ids = []
        for idx in range(args.device_count):
            device = (
                await client.post(
                    "/v1/devices",
                    json={"name": f"load-{idx}"},
                    headers={"Authorization": f"Bearer {args.api_token}"},
                )
            ).json()
            recording = (
                await client.post(
                    "/v1/recordings",
                    json={
                        "device_id": device["id"],
                        "started_at": start_ts.isoformat(),
                    },
                    headers={"Authorization": f"Bearer {args.api_token}"},
                )
            ).json()
            device_ids.append(device["id"])
            recording_ids.append(recording["id"])

        for device_index, recording_id in enumerate(recording_ids):
            for batch_start in range(0, total_epochs, args.ingest_batch_size):
                batch = generator.generate_epoch_batch(
                    device_index=device_index,
                    start_ts=start_ts,
                    epoch_seconds=args.epoch_seconds,
                    start_index=batch_start,
                    count=min(args.ingest_batch_size, total_epochs - batch_start),
                    total_epochs=total_epochs,
                    feature_schema_version="v1",
                )
                await client.post(
                    "/v1/epochs:ingest",
                    headers={"Authorization": f"Bearer {args.api_token}"},
                    json={"recording_id": recording_id, "epochs": batch},
                )

        semaphore = asyncio.Semaphore(args.predict_concurrency)

        async def _predict(recording_id: str) -> None:
            async with semaphore:
                await client.post(
                    "/v1/predict",
                    headers={"Authorization": f"Bearer {args.api_token}"},
                    json={"recording_id": recording_id},
                )

        predict_task = asyncio.gather(*[_predict(rid) for rid in recording_ids])

        if args.reload_during:
            await client.post(
                "/v1/models/reload",
                headers={"Authorization": f"Bearer {args.admin_token}"},
            )

        await predict_task

        perf = await client.get(
            "/internal/performance",
            headers={"Authorization": f"Bearer {args.admin_token}"},
        )
        print(perf.json())


if __name__ == "__main__":
    asyncio.run(main())
