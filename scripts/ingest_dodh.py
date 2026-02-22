from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid

from sqlalchemy.dialects.postgresql import insert

from app.core.settings import get_settings
from app.db.models import Device, Prediction, Recording, FeatureSchema
from app.db.session import run_with_db_retry
import json

from app.feature_store.service import get_feature_schema_by_version, register_feature_schema
from app.feature_store.schema import FeatureSchemaRecord
from app.services.ingest import ingest_epochs
from extractor.cli import find_h5_files
from extractor.config import ExtractConfig, FEATURE_KEYS
from extractor.extract import extract_recording

LABEL_MAP = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DODH data into Noctis DB")
    parser.add_argument("--input", required=True, help="Path to DODH dataset directory")
    parser.add_argument("--nights", type=int, default=20, help="Number of nights to ingest")
    parser.add_argument(
        "--feature-schema",
        default="configs/feature_schema_dodh.json",
        help="Feature schema JSON path",
    )
    parser.add_argument("--device-external-id", default="dodh", help="Device external id")
    parser.add_argument("--model-version", default="ground_truth", help="Prediction model tag")
    parser.add_argument("--epoch-sec", type=int, default=30)
    parser.add_argument(
        "--force-schema",
        action="store_true",
        default=False,
        help="Deactivate active schema to register new version",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    input_path = Path(args.input)
    schema_path = Path(args.feature_schema)
    files = find_h5_files(input_path)
    if not files:
        raise SystemExit("No .h5 files found")
    files = files[: max(0, int(args.nights))]
    config = ExtractConfig(epoch_sec=int(args.epoch_sec))

    payload = json.loads(schema_path.read_text())
    schema_version = str(payload.get("version"))

    def _ensure_schema(session):
        existing = get_feature_schema_by_version(session, schema_version)
        if existing:
            return existing
        if args.force_schema:
            session.query(FeatureSchema).filter(FeatureSchema.is_active.is_(True)).update(
                {FeatureSchema.is_active: False}
            )
        return register_feature_schema(session, payload=payload, activate=False)

    schema = run_with_db_retry(
        _ensure_schema,
        commit=True,
        operation_name="ensure_feature_schema",
    )

    def _ensure_device(session):
        device = (
            session.query(Device)
            .filter(Device.tenant_id == uuid.UUID(settings.default_tenant_id))
            .filter(Device.external_id == args.device_external_id)
            .one_or_none()
        )
        if device:
            return device
        device = Device(
            tenant_id=uuid.UUID(settings.default_tenant_id),
            name=args.device_external_id,
            external_id=args.device_external_id,
        )
        session.add(device)
        session.flush()
        return device

    device = run_with_db_retry(
        _ensure_device,
        commit=True,
        operation_name="ensure_device",
    )

    for path in files:
        result = extract_recording(path, config)
        started_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        ended_at = started_at + timedelta(seconds=len(result.stages) * config.epoch_sec)
        recording_id = uuid.uuid4()

        def _insert_recording(session):
            recording = Recording(
                id=recording_id,
                tenant_id=uuid.UUID(settings.default_tenant_id),
                device_id=device.id,
                started_at=started_at,
                ended_at=ended_at,
            )
            session.add(recording)
            session.flush()
            return recording

        recording = run_with_db_retry(
            _insert_recording,
            commit=True,
            operation_name="insert_recording",
        )

        epoch_rows: list[dict] = []
        prediction_rows: list[dict] = []
        feature_names = _feature_names(schema)
        for idx, stage in enumerate(result.stages):
            epoch_start_ts = started_at + timedelta(seconds=idx * config.epoch_sec)
            features = result.records[idx].features
            feature_vector = [float(features[name]) for name in feature_names]
            epoch_rows.append(
                {
                    "tenant_id": recording.tenant_id,
                    "recording_id": recording.id,
                    "epoch_index": idx,
                    "epoch_start_ts": epoch_start_ts,
                    "feature_schema_version": schema.version,
                    "features_payload": {"features": feature_vector},
                }
            )
            if not result.stage_known[idx]:
                continue
            label = LABEL_MAP.get(int(stage))
            if label is None:
                continue
            window_end_ts = epoch_start_ts + timedelta(seconds=config.epoch_sec)
            probabilities = {key: 0.0 for key in LABEL_MAP.values()}
            probabilities[label] = 1.0
            prediction_rows.append(
                {
                    "tenant_id": recording.tenant_id,
                    "recording_id": recording.id,
                    "window_start_ts": epoch_start_ts,
                    "window_end_ts": window_end_ts,
                    "model_version": args.model_version,
                    "feature_schema_version": schema.version,
                    "predicted_stage": label,
                    "ground_truth_stage": label,
                    "probabilities": probabilities,
                    "confidence": 1.0,
                }
            )

        def _insert_epochs(session):
            return ingest_epochs(session, epoch_rows)

        inserted_epochs = run_with_db_retry(
            _insert_epochs,
            commit=True,
            operation_name="ingest_epochs",
        )

        if prediction_rows:

            def _insert_predictions(session):
                stmt = insert(Prediction).values(prediction_rows)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=[Prediction.recording_id, Prediction.window_end_ts]
                )
                result = session.execute(stmt.returning(Prediction.id))
                return len(result.fetchall())

            inserted_predictions = run_with_db_retry(
                _insert_predictions,
                commit=True,
                operation_name="ingest_predictions",
            )
        else:
            inserted_predictions = 0

        print(f"recording={path.name} epochs={inserted_epochs} predictions={inserted_predictions}")


def _feature_names(schema: FeatureSchemaRecord) -> list[str]:
    if schema.feature_names:
        return schema.feature_names
    return FEATURE_KEYS


if __name__ == "__main__":
    main()
