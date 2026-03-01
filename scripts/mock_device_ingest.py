from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import random
import sys
from pathlib import Path
from urllib import error, request

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ml.feature_schema import load_feature_schema

DEFAULT_API_URL = os.getenv("NOCTIS_BASE_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("NOCTIS_API_KEY", "changeme")
DEFAULT_FEATURE_SCHEMA_PATH = ROOT_DIR / "models" / "active" / "feature_schema.json"

DREEM_V1_FEATURES = [
    "in_bed_pct",
    "hr_mean",
    "hr_std",
    "dhr",
    "rr_mean",
    "rr_std",
    "drr",
    "large_move_pct",
    "minor_move_pct",
    "turnovers_delta",
    "apnea_delta",
    "flags",
    "vib_move_pct",
    "vib_resp_q",
    "agree_flags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mock hardware device ingestion - sends realistic epoch data to backend"
    )
    parser.add_argument(
        "--device-external-id",
        required=True,
        help="Unique device identifier",
    )
    parser.add_argument(
        "--device-name",
        default=None,
        help="Device display name (required for first-time device registration)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of epochs to generate (default: 100)",
    )
    parser.add_argument(
        "--epoch-duration-seconds",
        type=int,
        default=30,
        help="Duration of each epoch in seconds (default: 30)",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Backend API URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help=f"API key for authentication (default: from NOCTIS_API_KEY env or 'changeme')",
    )
    parser.add_argument(
        "--forward-to-ml",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Whether to forward data to ML pipeline (default: true, use --forward-to-ml=false to disable)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducible data generation",
    )
    parser.add_argument(
        "--feature-schema-path",
        type=Path,
        default=DEFAULT_FEATURE_SCHEMA_PATH,
        help="Path to feature schema JSON (default: models/active/feature_schema.json)",
    )
    return parser.parse_args()


def generate_realistic_epoch(
    epoch_index: int,
    base_time: datetime,
    epoch_duration_seconds: int,
    feature_names: list[str],
    rng: random.Random,
) -> dict:
    in_bed = rng.choice([0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0])

    hr_mean = rng.gauss(60, 15)
    hr_mean = max(40, min(100, hr_mean))
    hr_std = abs(rng.gauss(5, 3))
    dhr = rng.gauss(0, 2)

    rr_mean = rng.gauss(14, 3)
    rr_mean = max(8, min(25, rr_mean))
    rr_std = abs(rng.gauss(2, 1))
    drr = rng.gauss(0, 1)

    large_move_pct = rng.random() * 0.1
    minor_move_pct = rng.random() * 0.4
    vib_move_pct = rng.random() * 0.2

    turnovers_delta = rng.randint(-2, 5)
    apnea_delta = rng.randint(0, 3) if rng.random() < 0.1 else 0
    flags = 0.0
    vib_resp_q = rng.random() * 0.3
    agree_flags = rng.uniform(0.85, 1.0)

    epoch_time = base_time.replace(second=0, microsecond=0) + timedelta(
        seconds=epoch_index * epoch_duration_seconds
    )

    metrics = [
        in_bed,
        hr_mean,
        hr_std,
        dhr,
        rr_mean,
        rr_std,
        drr,
        large_move_pct,
        minor_move_pct,
        turnovers_delta,
        apnea_delta,
        flags,
        vib_move_pct,
        vib_resp_q,
        agree_flags,
    ]

    return {
        "epoch_index": epoch_index,
        "epoch_start_ts": epoch_time.isoformat().replace("+00:00", "Z"),
        "metrics": metrics,
    }


def load_feature_schema_version(path: Path) -> tuple[list[str], str]:
    if path.exists():
        schema = load_feature_schema(path)
        return list(schema.features), schema.version
    return DREEM_V1_FEATURES, "dreem_v1"


def build_payload(
    device_external_id: str,
    device_name: str | None,
    num_epochs: int,
    epoch_duration_seconds: int,
    forward_to_ml: bool,
    feature_names: list[str],
    feature_schema_version: str,
    rng: random.Random,
) -> dict:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    recording_started_at = now.isoformat().replace("+00:00", "Z")

    epochs = [
        generate_realistic_epoch(i, now, epoch_duration_seconds, feature_names, rng)
        for i in range(num_epochs)
    ]

    payload = {
        "device_external_id": device_external_id,
        "recording_started_at": recording_started_at,
        "forward_to_ml": forward_to_ml,
        "epochs": epochs,
    }

    if device_name:
        payload["device_name"] = device_name

    return payload


def post_device_ingest(api_url: str, api_key: str, payload: dict) -> dict:
    url = f"{api_url.rstrip('/')}/v1/epochs:ingest-device"
    body = json.dumps(payload).encode("utf-8")

    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        },
    )

    try:
        with request.urlopen(req, timeout=30) as response:
            data = response.read().decode("utf-8")
            return json.loads(data) if data else {}
    except error.HTTPError as exc:
        try:
            error_body = json.loads(exc.read().decode("utf-8"))
            detail = error_body.get("detail", str(exc))
        except Exception:
            detail = f"HTTP {exc.code}"
        raise SystemExit(f"Failed to ingest: {detail}")


def main() -> None:
    args = parse_args()

    if args.random_seed is not None:
        rng = random.Random(args.random_seed)
    else:
        rng = random.Random()

    feature_names, schema_version = load_feature_schema_version(args.feature_schema_path)

    if len(feature_names) != 15:
        print(
            f"WARNING: Expected 15 features, got {len(feature_names)}. "
            "Using default dreem_v1 schema.",
            file=sys.stderr,
        )
        feature_names = DREEM_V1_FEATURES

    print(
        f"Generating {args.num_epochs} epochs for device '{args.device_external_id}' "
        f"(schema: {schema_version}, features: {len(feature_names)})"
    )

    payload = build_payload(
        device_external_id=args.device_external_id,
        device_name=args.device_name,
        num_epochs=args.num_epochs,
        epoch_duration_seconds=args.epoch_duration_seconds,
        forward_to_ml=args.forward_to_ml,
        feature_names=feature_names,
        feature_schema_version=schema_version,
        rng=rng,
    )

    result = post_device_ingest(args.api_url, args.api_key, payload)

    print(f"Success:")
    print(f"  device_id: {result.get('device_id')}")
    print(f"  recording_id: {result.get('recording_id')}")
    print(f"  received: {result.get('received')}")
    print(f"  inserted: {result.get('inserted')}")
    print(f"  raw_inserted: {result.get('raw_inserted')}")
    print(f"  forwarded: {result.get('forwarded')}")


if __name__ == "__main__":
    main()
