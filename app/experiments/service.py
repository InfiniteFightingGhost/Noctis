from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import get_settings
from app.db.models import Experiment, ModelVersion, TrainingRun
from app.ml.feature_schema import load_feature_schema


def create_experiment_if_missing(
    session: Session, name: str, *, tenant_id: uuid.UUID | str | None = None
) -> Experiment:
    if tenant_id is None:
        tenant_id = get_settings().default_tenant_id
    if isinstance(tenant_id, str):
        tenant_id = uuid.UUID(tenant_id)
    stmt = (
        select(Experiment)
        .where(Experiment.name == name)
        .where(Experiment.tenant_id == tenant_id)
    )
    existing = session.execute(stmt).scalar_one_or_none()
    if existing:
        return existing
    experiment = Experiment(name=name, tenant_id=tenant_id)
    session.add(experiment)
    session.flush()
    return experiment


def register_model_version(
    session: Session,
    *,
    version: str,
    status: str,
    metrics: dict[str, Any],
    feature_schema_version: str | None = None,
    feature_schema_path: Path | None = None,
    artifact_path: str,
    dataset_snapshot_id: uuid.UUID | str | None = None,
    training_seed: int | None = None,
    git_commit_hash: str | None = None,
    metrics_hash: str | None = None,
    artifact_hash: str | None = None,
) -> ModelVersion:
    existing = session.execute(
        select(ModelVersion).where(ModelVersion.version == version)
    ).scalar_one_or_none()
    if existing:
        raise ValueError("Model version already registered")
    if feature_schema_version is None:
        if feature_schema_path is None:
            raise ValueError("feature_schema_version is required")
        schema = load_feature_schema(feature_schema_path)
        feature_schema_version = schema.version
    dataset_snapshot_uuid: uuid.UUID | None = None
    if dataset_snapshot_id:
        dataset_snapshot_uuid = (
            dataset_snapshot_id
            if isinstance(dataset_snapshot_id, uuid.UUID)
            else uuid.UUID(str(dataset_snapshot_id))
        )
    model_version = ModelVersion(
        version=version,
        status=status,
        metrics=metrics,
        feature_schema_version=feature_schema_version,
        dataset_snapshot_id=dataset_snapshot_uuid,
        training_seed=training_seed,
        git_commit_hash=git_commit_hash,
        metrics_hash=metrics_hash,
        artifact_hash=artifact_hash,
        artifact_path=artifact_path,
        details={"created_at": datetime.now(timezone.utc).isoformat()},
    )
    session.add(model_version)
    session.flush()
    return model_version


def register_training_run(
    session: Session,
    *,
    experiment_id: Any | None,
    model_version: str,
    status: str,
    metrics: dict[str, Any],
    hyperparameters: dict[str, Any],
    dataset_dir: str,
    feature_schema_version: str | None = None,
    feature_schema_path: Path | None = None,
    artifact_path: str,
    dataset_snapshot_id: uuid.UUID | str | None = None,
) -> TrainingRun:
    dataset_snapshot = _load_dataset_snapshot(Path(dataset_dir))
    if feature_schema_version is None:
        if feature_schema_path is None:
            raise ValueError("feature_schema_version is required")
        schema = load_feature_schema(feature_schema_path)
        feature_schema_version = schema.version
    dataset_snapshot_uuid: uuid.UUID | None = None
    if dataset_snapshot_id:
        dataset_snapshot_uuid = (
            dataset_snapshot_id
            if isinstance(dataset_snapshot_id, uuid.UUID)
            else uuid.UUID(str(dataset_snapshot_id))
        )
    run = TrainingRun(
        experiment_id=experiment_id,
        model_version=model_version,
        status=status,
        hyperparameters=hyperparameters,
        dataset_snapshot=dataset_snapshot,
        metrics=metrics,
        feature_schema_version=feature_schema_version,
        commit_hash=_read_commit_hash(dataset_snapshot),
        artifact_path=artifact_path,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    session.add(run)
    session.flush()
    if dataset_snapshot_uuid:
        session.query(ModelVersion).filter(
            ModelVersion.version == model_version
        ).update({ModelVersion.dataset_snapshot_id: dataset_snapshot_uuid})
    session.query(ModelVersion).filter(ModelVersion.version == model_version).update(
        {ModelVersion.training_run_id: run.id}
    )
    return run


def list_experiments(session: Session, *, tenant_id: uuid.UUID) -> list[dict[str, Any]]:
    experiments = (
        session.execute(select(Experiment).where(Experiment.tenant_id == tenant_id))
        .scalars()
        .all()
    )
    return [
        {
            "id": str(item.id),
            "name": item.name,
            "description": item.description,
            "created_at": item.created_at,
        }
        for item in experiments
    ]


def list_models(session: Session) -> list[dict[str, Any]]:
    models = session.execute(select(ModelVersion)).scalars().all()
    return [
        {
            "version": model.version,
            "status": model.status,
            "metrics": model.metrics,
            "feature_schema_version": model.feature_schema_version,
            "dataset_snapshot_id": model.dataset_snapshot_id,
            "training_run_id": model.training_run_id,
            "git_commit_hash": model.git_commit_hash,
            "training_seed": model.training_seed,
            "metrics_hash": model.metrics_hash,
            "artifact_hash": model.artifact_hash,
            "artifact_path": model.artifact_path,
            "created_at": model.created_at,
            "promoted_at": model.promoted_at,
            "promoted_by": model.promoted_by,
            "deployed_at": model.deployed_at,
        }
        for model in models
    ]


def get_model(session: Session, version: str) -> dict[str, Any] | None:
    model = session.execute(
        select(ModelVersion).where(ModelVersion.version == version)
    ).scalar_one_or_none()
    if model is None:
        return None
    return {
        "version": model.version,
        "status": model.status,
        "metrics": model.metrics,
        "feature_schema_version": model.feature_schema_version,
        "dataset_snapshot_id": model.dataset_snapshot_id,
        "training_run_id": model.training_run_id,
        "git_commit_hash": model.git_commit_hash,
        "training_seed": model.training_seed,
        "metrics_hash": model.metrics_hash,
        "artifact_hash": model.artifact_hash,
        "artifact_path": model.artifact_path,
        "created_at": model.created_at,
        "promoted_at": model.promoted_at,
        "promoted_by": model.promoted_by,
        "deployed_at": model.deployed_at,
        "details": model.details,
    }


def _load_dataset_snapshot(dataset_dir: Path) -> dict[str, Any] | None:
    path = dataset_dir / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_commit_hash(snapshot: dict[str, Any] | None) -> str | None:
    if not snapshot:
        return None
    return snapshot.get("git_commit_hash")
