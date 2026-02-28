from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.settings import get_settings
from app.db.models import Device, FeatureSchema, Prediction, Recording
from app.db.session import run_with_db_retry
from app.feature_store.service import get_feature_schema_by_version, register_feature_schema
from app.services.ingest import ingest_epochs
from dreem_extractor.config import load_config as load_dreem_config
from dreem_extractor.constants import FEATURE_ORDER
from dreem_extractor.pipeline import extract_record as extract_dreem_record
from dreem_extractor.serialize.writers import (
    write_manifest as write_dreem_manifest,
    write_record_outputs as write_dreem_outputs,
)
from edf_extractor.config import load_config as load_edf_config
from edf_extractor.pipeline import extract_record as extract_edf_record
from edf_extractor.serialize.writers import (
    write_manifest as write_edf_manifest,
    write_record_outputs as write_edf_outputs,
)
from extractor_hardened.errors import to_failure_payload

LABEL_MAP = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


@dataclass
class OutputRow:
    dataset: str
    record_id: str
    source_path: str
    hypnogram_path: str | None
    artifact_dir: str | None
    failure_path: str | None
    n_epochs: int
    valid_ratio: float
    recording_db_id: str | None


@dataclass
class DBContext:
    tenant_id: uuid.UUID
    feature_schema_version: str
    model_version: str
    device_prefix: str


FAILURE_COUNTS: Counter[str] = Counter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DODH/CAP/ISRUC and save to DB/files")
    parser.add_argument(
        "--datasets-root",
        default="../../Datasets",
        help="Root folder containing dataset subdirectories",
    )
    parser.add_argument(
        "--output",
        default="artifacts/dataset_extracts",
        help="Output directory for extracted records and indexes",
    )
    parser.add_argument(
        "--sink",
        choices=("db", "files", "both"),
        default="db",
        help="Where to save extracted data",
    )
    parser.add_argument(
        "--feature-schema-version",
        default="dreem_v1",
        help="Feature schema version for DB ingest",
    )
    parser.add_argument(
        "--model-version",
        default="ground_truth",
        help="Prediction model tag used for labels",
    )
    parser.add_argument(
        "--device-prefix",
        default="dataset",
        help="Device external-id prefix for DB ingest",
    )
    parser.add_argument(
        "--force-schema",
        action="store_true",
        default=False,
        help="Deactivate active schema when registering DB schema",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=0,
        help="Optional max record count per dataset (0 means no limit)",
    )
    parser.add_argument(
        "--isruc-scorer",
        type=int,
        choices=(1, 2),
        default=1,
        help="Preferred scorer file for ISRUC (_1.txt or _2.txt)",
    )
    parser.add_argument(
        "--single-file",
        default=None,
        help="Extract a single file only (.h5/.edf/.rec)",
    )
    parser.add_argument(
        "--single-dataset",
        choices=("dodh", "cap", "isruc"),
        default=None,
        help="Dataset type for --single-file",
    )
    parser.add_argument(
        "--single-hypnogram",
        default=None,
        help="Optional explicit hypnogram path for --single-file EDF/REC",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets_root = Path(args.datasets_root).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    write_files = args.sink in ("files", "both")
    write_db = args.sink in ("db", "both")
    db_ctx = _prepare_db(args) if write_db else None

    if args.single_file:
        if not args.single_dataset:
            raise SystemExit("--single-dataset is required with --single-file")
        row = _extract_single(
            dataset=args.single_dataset,
            source_path=Path(args.single_file).resolve(),
            output_root=output_root,
            preferred_scorer=args.isruc_scorer,
            hypnogram_override=Path(args.single_hypnogram).resolve()
            if args.single_hypnogram
            else None,
            write_files=write_files,
            write_db=write_db,
            db_ctx=db_ctx,
        )
        rows = [row]
        _write_training_index(rows, output_root)
        _write_summary(rows, output_root)
        print(f"Extracted records: {len(rows)}")
        if FAILURE_COUNTS:
            print(f"Failure counts by code: {dict(FAILURE_COUNTS)}")
        print(f"Output root: {output_root}")
        return

    rows: list[OutputRow] = []
    rows.extend(
        _extract_dodh(
            datasets_root,
            output_root,
            args.limit_per_dataset,
            write_files=write_files,
            write_db=write_db,
            db_ctx=db_ctx,
        )
    )
    rows.extend(
        _extract_cap(
            datasets_root,
            output_root,
            args.limit_per_dataset,
            write_files=write_files,
            write_db=write_db,
            db_ctx=db_ctx,
        )
    )
    rows.extend(
        _extract_isruc(
            datasets_root,
            output_root,
            args.limit_per_dataset,
            preferred_scorer=args.isruc_scorer,
            write_files=write_files,
            write_db=write_db,
            db_ctx=db_ctx,
        )
    )

    _write_training_index(rows, output_root)
    _write_summary(rows, output_root)
    _warn_archived_datasets(datasets_root)
    print(f"Extracted records: {len(rows)}")
    if FAILURE_COUNTS:
        print(f"Failure counts by code: {dict(FAILURE_COUNTS)}")
    print(f"Output root: {output_root}")


def _prepare_db(args: argparse.Namespace) -> DBContext:
    settings = get_settings()
    tenant_id = uuid.UUID(settings.default_tenant_id)
    payload = _schema_payload(args.feature_schema_version)

    def _ensure_schema(session):
        existing = get_feature_schema_by_version(session, args.feature_schema_version)
        if existing:
            return existing
        if args.force_schema:
            session.query(FeatureSchema).filter(FeatureSchema.is_active.is_(True)).update(
                {FeatureSchema.is_active: False}
            )
        return register_feature_schema(session, payload=payload, activate=False)

    run_with_db_retry(
        _ensure_schema,
        commit=True,
        operation_name="ensure_all_dataset_feature_schema",
    )
    return DBContext(
        tenant_id=tenant_id,
        feature_schema_version=args.feature_schema_version,
        model_version=args.model_version,
        device_prefix=args.device_prefix,
    )


def _extract_dodh(
    datasets_root: Path,
    output_root: Path,
    limit: int,
    *,
    write_files: bool,
    write_db: bool,
    db_ctx: DBContext | None,
) -> list[OutputRow]:
    dodh_root = datasets_root / "dodh"
    files = sorted(
        path
        for path in dodh_root.rglob("*.h5")
        if "__MACOSX" not in str(path) and "features_out" not in str(path)
    )
    if limit > 0:
        files = files[:limit]
    if not files:
        return []

    config = load_dreem_config()
    out_dir = output_root / "dodh"
    if write_files:
        out_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, object]] = []
    failures_dir = out_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[OutputRow] = []

    for path in files:
        try:
            result = extract_dreem_record(path, config)
        except Exception as exc:
            failure_path = _write_failure_file("dodh", path, failures_dir, exc)
            rows.append(
                OutputRow(
                    dataset="dodh",
                    record_id=path.stem,
                    source_path=str(path),
                    hypnogram_path="/hypnogram",
                    artifact_dir=None,
                    failure_path=str(failure_path),
                    n_epochs=0,
                    valid_ratio=0.0,
                    recording_db_id=None,
                )
            )
            continue

        artifact_dir: str | None = None
        if write_files:
            outputs = write_dreem_outputs(result, out_dir)
            manifest_entries.append(
                {
                    "record_id": result.record_id,
                    "record_dir": str(outputs["record_dir"]),
                }
            )
            artifact_dir = str(outputs["record_dir"])

        recording_db_id = None
        if write_db and db_ctx is not None:
            recording_db_id = _ingest_result(
                dataset="dodh",
                source_path=path,
                hypnogram_path="/hypnogram",
                hypnogram=result.hypnogram,
                features=result.features,
                epoch_sec=config.epoch_sec,
                metadata=result.metadata,
                db_ctx=db_ctx,
            )

        rows.append(
            OutputRow(
                dataset="dodh",
                record_id=result.record_id,
                source_path=str(path),
                hypnogram_path="/hypnogram",
                artifact_dir=artifact_dir,
                failure_path=None,
                n_epochs=int(result.hypnogram.shape[0]),
                valid_ratio=float(np.mean(result.valid_mask)) if result.valid_mask.size else 0.0,
                recording_db_id=recording_db_id,
            )
        )

    if write_files:
        write_dreem_manifest(manifest_entries, out_dir / "manifest.jsonl")
    return rows


def _extract_cap(
    datasets_root: Path,
    output_root: Path,
    limit: int,
    *,
    write_files: bool,
    write_db: bool,
    db_ctx: DBContext | None,
) -> list[OutputRow]:
    cap_root = datasets_root / "capslpdb-1.0.0"
    files = sorted(cap_root.glob("*.edf"))
    if limit > 0:
        files = files[:limit]
    if not files:
        return []

    config = load_edf_config()
    out_dir = output_root / "cap"
    if write_files:
        out_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, object]] = []
    failures_dir = out_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[OutputRow] = []

    for edf_path in files:
        hyp_path = cap_root / f"{edf_path.stem}.txt"
        if not hyp_path.exists():
            failure_path = _write_failure_file(
                "cap",
                edf_path,
                failures_dir,
                FileNotFoundError(f"Missing hypnogram: {hyp_path}"),
            )
            rows.append(
                OutputRow(
                    dataset="cap",
                    record_id=edf_path.stem,
                    source_path=str(edf_path),
                    hypnogram_path=str(hyp_path),
                    artifact_dir=None,
                    failure_path=str(failure_path),
                    n_epochs=0,
                    valid_ratio=0.0,
                    recording_db_id=None,
                )
            )
            continue
        try:
            result = extract_edf_record(edf_path, hyp_path, config)
        except Exception as exc:
            failure_path = _write_failure_file("cap", edf_path, failures_dir, exc)
            rows.append(
                OutputRow(
                    dataset="cap",
                    record_id=edf_path.stem,
                    source_path=str(edf_path),
                    hypnogram_path=str(hyp_path),
                    artifact_dir=None,
                    failure_path=str(failure_path),
                    n_epochs=0,
                    valid_ratio=0.0,
                    recording_db_id=None,
                )
            )
            continue

        artifact_dir: str | None = None
        if write_files:
            outputs = write_edf_outputs(result, out_dir)
            manifest_entries.append(
                {
                    "record_id": result.record_id,
                    "record_dir": str(outputs["record_dir"]),
                }
            )
            artifact_dir = str(outputs["record_dir"])

        recording_db_id = None
        if write_db and db_ctx is not None:
            recording_db_id = _ingest_result(
                dataset="cap",
                source_path=edf_path,
                hypnogram_path=str(hyp_path),
                hypnogram=result.hypnogram,
                features=result.features,
                epoch_sec=config.epoch_sec,
                metadata=result.metadata,
                db_ctx=db_ctx,
            )

        rows.append(
            OutputRow(
                dataset="cap",
                record_id=result.record_id,
                source_path=str(edf_path),
                hypnogram_path=str(hyp_path),
                artifact_dir=artifact_dir,
                failure_path=None,
                n_epochs=int(result.hypnogram.shape[0]),
                valid_ratio=float(np.mean(result.valid_mask)) if result.valid_mask.size else 0.0,
                recording_db_id=recording_db_id,
            )
        )

    if write_files:
        write_edf_manifest(manifest_entries, out_dir / "manifest.jsonl")
    return rows


def _extract_isruc(
    datasets_root: Path,
    output_root: Path,
    limit: int,
    *,
    preferred_scorer: int,
    write_files: bool,
    write_db: bool,
    db_ctx: DBContext | None,
) -> list[OutputRow]:
    isruc_root = datasets_root / "isruc" / "raw"
    files = sorted(isruc_root.rglob("*.rec"))
    if limit > 0:
        files = files[:limit]
    if not files:
        return []

    config = load_edf_config()
    out_dir = output_root / "isruc"
    if write_files:
        out_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, object]] = []
    failures_dir = out_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[OutputRow] = []

    for rec_path in files:
        hyp_path = _resolve_isruc_hypnogram(rec_path, preferred_scorer)
        if hyp_path is None:
            failure_path = _write_failure_file(
                "isruc",
                rec_path,
                failures_dir,
                FileNotFoundError("Missing ISRUC scorer file"),
            )
            rows.append(
                OutputRow(
                    dataset="isruc",
                    record_id=rec_path.stem,
                    source_path=str(rec_path),
                    hypnogram_path=None,
                    artifact_dir=None,
                    failure_path=str(failure_path),
                    n_epochs=0,
                    valid_ratio=0.0,
                    recording_db_id=None,
                )
            )
            continue
        try:
            result = extract_edf_record(rec_path, hyp_path, config)
        except Exception as exc:
            failure_path = _write_failure_file("isruc", rec_path, failures_dir, exc)
            rows.append(
                OutputRow(
                    dataset="isruc",
                    record_id=rec_path.stem,
                    source_path=str(rec_path),
                    hypnogram_path=str(hyp_path),
                    artifact_dir=None,
                    failure_path=str(failure_path),
                    n_epochs=0,
                    valid_ratio=0.0,
                    recording_db_id=None,
                )
            )
            continue

        artifact_dir: str | None = None
        if write_files:
            outputs = write_edf_outputs(result, out_dir)
            manifest_entries.append(
                {
                    "record_id": result.record_id,
                    "record_dir": str(outputs["record_dir"]),
                }
            )
            artifact_dir = str(outputs["record_dir"])

        recording_db_id = None
        if write_db and db_ctx is not None:
            recording_db_id = _ingest_result(
                dataset="isruc",
                source_path=rec_path,
                hypnogram_path=str(hyp_path),
                hypnogram=result.hypnogram,
                features=result.features,
                epoch_sec=config.epoch_sec,
                metadata=result.metadata,
                db_ctx=db_ctx,
            )

        rows.append(
            OutputRow(
                dataset="isruc",
                record_id=result.record_id,
                source_path=str(rec_path),
                hypnogram_path=str(hyp_path),
                artifact_dir=artifact_dir,
                failure_path=None,
                n_epochs=int(result.hypnogram.shape[0]),
                valid_ratio=float(np.mean(result.valid_mask)) if result.valid_mask.size else 0.0,
                recording_db_id=recording_db_id,
            )
        )

    if write_files:
        write_edf_manifest(manifest_entries, out_dir / "manifest.jsonl")
    return rows


def _extract_single(
    *,
    dataset: str,
    source_path: Path,
    output_root: Path,
    preferred_scorer: int,
    hypnogram_override: Path | None,
    write_files: bool,
    write_db: bool,
    db_ctx: DBContext | None,
) -> OutputRow:
    failures_dir = output_root / dataset / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)

    try:
        if dataset == "dodh":
            config = load_dreem_config()
            result = extract_dreem_record(source_path, config)
            hyp_path = "/hypnogram"
            write_outputs = write_dreem_outputs
            write_manifest = write_dreem_manifest
        elif dataset in {"cap", "isruc"}:
            config = load_edf_config()
            if hypnogram_override is not None:
                hyp_path = hypnogram_override
            elif dataset == "isruc":
                resolved = _resolve_isruc_hypnogram(source_path, preferred_scorer)
                if resolved is None:
                    raise FileNotFoundError("Missing ISRUC scorer file")
                hyp_path = resolved
            else:
                hyp_path = source_path.with_suffix(".txt")
            if not Path(hyp_path).exists():
                raise FileNotFoundError(f"Missing hypnogram: {hyp_path}")
            result = extract_edf_record(source_path, hyp_path, config)
            write_outputs = write_edf_outputs
            write_manifest = write_edf_manifest
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
    except Exception as exc:
        failure_path = _write_failure_file(dataset, source_path, failures_dir, exc)
        return OutputRow(
            dataset=dataset,
            record_id=source_path.stem,
            source_path=str(source_path),
            hypnogram_path=str(hypnogram_override) if hypnogram_override else None,
            artifact_dir=None,
            failure_path=str(failure_path),
            n_epochs=0,
            valid_ratio=0.0,
            recording_db_id=None,
        )

    out_dir = output_root / dataset
    artifact_dir: str | None = None
    if write_files:
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = write_outputs(result, out_dir)
        write_manifest(
            [{"record_id": result.record_id, "record_dir": str(outputs["record_dir"])}],
            out_dir / "manifest.jsonl",
        )
        artifact_dir = str(outputs["record_dir"])

    recording_db_id = None
    if write_db and db_ctx is not None:
        recording_db_id = _ingest_result(
            dataset=dataset,
            source_path=source_path,
            hypnogram_path=str(hyp_path),
            hypnogram=result.hypnogram,
            features=result.features,
            epoch_sec=config.epoch_sec,
            metadata=result.metadata,
            db_ctx=db_ctx,
        )

    return OutputRow(
        dataset=dataset,
        record_id=result.record_id,
        source_path=str(source_path),
        hypnogram_path=str(hyp_path),
        artifact_dir=artifact_dir,
        failure_path=None,
        n_epochs=int(result.hypnogram.shape[0]),
        valid_ratio=float(np.mean(result.valid_mask)) if result.valid_mask.size else 0.0,
        recording_db_id=recording_db_id,
    )


def _ingest_result(
    *,
    dataset: str,
    source_path: Path,
    hypnogram_path: str | None,
    hypnogram: np.ndarray,
    features: np.ndarray,
    epoch_sec: int,
    metadata: dict[str, object],
    db_ctx: DBContext,
) -> str:
    device_external_id = f"{db_ctx.device_prefix}-{dataset}"
    device = _ensure_device(db_ctx.tenant_id, device_external_id)
    started_at = _resolve_started_at(source_path, metadata.get("start_time"))
    ended_at = started_at + timedelta(seconds=int(len(hypnogram)) * epoch_sec)
    recording = _insert_recording(db_ctx.tenant_id, device.id, started_at, ended_at)

    epoch_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for idx, stage_raw in enumerate(hypnogram.astype(int)):
        epoch_start_ts = started_at + timedelta(seconds=idx * epoch_sec)
        feature_vector = [float(v) for v in features[idx, : len(FEATURE_ORDER)].tolist()]
        epoch_rows.append(
            {
                "tenant_id": db_ctx.tenant_id,
                "recording_id": recording.id,
                "epoch_index": idx,
                "epoch_start_ts": epoch_start_ts,
                "feature_schema_version": db_ctx.feature_schema_version,
                "features_payload": {"features": feature_vector},
            }
        )
        label = LABEL_MAP.get(stage_raw)
        if label is None:
            continue
        window_end_ts = epoch_start_ts + timedelta(seconds=epoch_sec)
        probs = {key: 0.0 for key in LABEL_MAP.values()}
        probs[label] = 1.0
        prediction_rows.append(
            {
                "tenant_id": db_ctx.tenant_id,
                "recording_id": recording.id,
                "window_start_ts": epoch_start_ts,
                "window_end_ts": window_end_ts,
                "model_version": db_ctx.model_version,
                "feature_schema_version": db_ctx.feature_schema_version,
                "predicted_stage": label,
                "ground_truth_stage": label,
                "probabilities": probs,
                "confidence": 1.0,
            }
        )

    run_with_db_retry(
        lambda session: ingest_epochs(session, epoch_rows),
        commit=True,
        operation_name="ingest_all_epochs",
    )

    if prediction_rows:
        run_with_db_retry(
            lambda session: _insert_predictions(session, prediction_rows),
            commit=True,
            operation_name="ingest_all_predictions",
        )

    print(
        f"ingested dataset={dataset} recording={recording.id} "
        f"epochs={len(epoch_rows)} predictions={len(prediction_rows)} source={source_path.name}"
    )
    return str(recording.id)


def _ensure_device(tenant_id: uuid.UUID, external_id: str) -> Device:
    def _op(session):
        existing = (
            session.query(Device)
            .filter(Device.tenant_id == tenant_id)
            .filter(Device.external_id == external_id)
            .one_or_none()
        )
        if existing is not None:
            return existing
        device = Device(tenant_id=tenant_id, name=external_id, external_id=external_id)
        session.add(device)
        session.flush()
        return device

    return run_with_db_retry(_op, commit=True, operation_name="ensure_dataset_device")


def _insert_recording(
    tenant_id: uuid.UUID,
    device_id: uuid.UUID,
    started_at: datetime,
    ended_at: datetime,
) -> Recording:
    recording_id = uuid.uuid4()

    def _op(session):
        recording = Recording(
            id=recording_id,
            tenant_id=tenant_id,
            device_id=device_id,
            started_at=started_at,
            ended_at=ended_at,
        )
        session.add(recording)
        session.flush()
        return recording

    return run_with_db_retry(_op, commit=True, operation_name="insert_dataset_recording")


def _insert_predictions(session, rows: list[dict]) -> int:
    stmt = insert(Prediction).values(rows)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=[Prediction.recording_id, Prediction.window_end_ts]
    )
    result = session.execute(stmt.returning(Prediction.id))
    return len(result.fetchall())


def _resolve_started_at(path: Path, start_time_value: object) -> datetime:
    if isinstance(start_time_value, str):
        value = start_time_value.strip()
        if value:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except ValueError:
                pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _schema_payload(version: str) -> dict[str, object]:
    features = [
        {
            "name": name,
            "dtype": "float32",
            "allowed_range": None,
            "description": None,
            "introduced_in_version": version,
            "deprecated_in_version": None,
            "position": idx,
        }
        for idx, name in enumerate(FEATURE_ORDER)
    ]
    return {
        "version": version,
        "description": "Dreem-compatible extractor schema for DODH/CAP/ISRUC",
        "features": features,
    }


def _resolve_isruc_hypnogram(rec_path: Path, preferred_scorer: int) -> Path | None:
    stem = rec_path.stem
    scorers = [preferred_scorer, 2 if preferred_scorer == 1 else 1]
    for scorer in scorers:
        candidate = rec_path.parent / f"{stem}_{scorer}.txt"
        if candidate.exists():
            return candidate
    return None


def _write_failure_file(
    dataset: str, source_path: Path, failures_dir: Path, exc: Exception
) -> Path:
    failures_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "source_path": str(source_path),
    }
    payload.update(to_failure_payload(exc))
    error_code = str(payload.get("error_code", "E_CONTRACT_VIOLATION"))
    FAILURE_COUNTS[error_code] += 1
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    failure_name = f"{source_path.stem}_{digest}.failure.json"
    failure_path = failures_dir / failure_name
    with failure_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return failure_path


def _write_training_index(rows: list[OutputRow], output_root: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "dataset": row.dataset,
                "record_id": row.record_id,
                "source_path": row.source_path,
                "hypnogram_path": row.hypnogram_path,
                "artifact_dir": row.artifact_dir,
                "failure_path": row.failure_path,
                "recording_db_id": row.recording_db_id,
                "n_epochs": row.n_epochs,
                "valid_ratio": row.valid_ratio,
            }
            for row in rows
        ]
    )
    csv_path = output_root / "training_index.csv"
    parquet_path = output_root / "training_index.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)


def _write_summary(rows: list[OutputRow], output_root: Path) -> None:
    by_dataset: dict[str, dict[str, float | int]] = {}
    for row in rows:
        bucket = by_dataset.setdefault(
            row.dataset,
            {"records": 0, "epochs": 0, "valid_ratio_sum": 0.0},
        )
        bucket["records"] = int(bucket["records"]) + 1
        bucket["epochs"] = int(bucket["epochs"]) + row.n_epochs
        bucket["valid_ratio_sum"] = float(bucket["valid_ratio_sum"]) + row.valid_ratio

    summary = {
        "datasets": {},
        "total_records": len(rows),
        "error_code_counts": dict(FAILURE_COUNTS),
    }
    for name, data in by_dataset.items():
        records = int(data["records"])
        summary["datasets"][name] = {
            "records": records,
            "epochs": int(data["epochs"]),
            "avg_valid_ratio": float(data["valid_ratio_sum"]) / records if records else 0.0,
        }

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def _warn_archived_datasets(datasets_root: Path) -> None:
    for zip_path in sorted(datasets_root.rglob("*.zip")):
        print(f"[warn] archive not extracted, skipping: {zip_path}")


if __name__ == "__main__":
    main()
