from __future__ import annotations

import time
from datetime import datetime, timezone

from app.core.settings import get_settings
from app.dataset.snapshot_config import snapshot_config_from_payload
from app.dataset.snapshots import create_snapshot
from app.db.session import run_with_db_retry
from app.experiments.service import (
    register_model_version,
    register_training_run,
)
from app.audit.scheduler import run_audit_cycle
from app.scheduler.service import (
    fetch_pending_jobs,
    mark_job_completed,
    mark_job_running,
)
from app.training.config import training_config_from_payload
from app.training.trainer import train_model
from app.training.versioning import next_version


def run_scheduler_loop() -> None:
    settings = get_settings()
    interval = settings.retrain_poll_interval_seconds
    audit_interval = settings.audit_poll_interval_seconds
    last_audit_at = datetime.min.replace(tzinfo=timezone.utc)
    while True:
        process_pending_jobs()
        now = datetime.now(timezone.utc)
        if (now - last_audit_at).total_seconds() >= audit_interval:
            run_audit_cycle()
            last_audit_at = now
        time.sleep(interval)


def process_pending_jobs() -> None:
    settings = get_settings()

    def _fetch(session):
        return fetch_pending_jobs(session, settings.retrain_batch_size)

    jobs = run_with_db_retry(_fetch, operation_name="fetch_retrain_jobs")
    for job in jobs:
        _process_job(job.id)


def _process_job(job_id) -> None:
    def _mark_running(session):
        job = _get_job(session, job_id)
        mark_job_running(session, job)
        return job

    job = run_with_db_retry(
        _mark_running, commit=True, operation_name="mark_retrain_running"
    )
    model_version = None
    error_message = None
    try:
        dataset_payload = job.dataset_config or {}
        snapshot_config = snapshot_config_from_payload(dataset_payload)
        snapshot_result = create_snapshot(snapshot_config)
        training_payload = dict(job.training_config or {})
        training_payload["dataset_dir"] = str(snapshot_config.output_dir)
        training_payload["dataset_snapshot_id"] = str(snapshot_result.snapshot_id)
        training_config = training_config_from_payload(training_payload)
        version = run_with_db_retry(
            lambda session: next_version(
                session=session,
                output_root=training_config.output_root,
                bump=training_config.version_bump,
            ),
            operation_name="next_model_version",
        )
        result = train_model(config=training_config, version=version)

        def _register(session):
            register_model_version(
                session,
                version=result.version,
                status="staging",
                metrics=result.metrics,
                feature_schema_version=result.feature_schema_version,
                artifact_path=str(result.artifact_dir),
                dataset_snapshot_id=result.dataset_snapshot_id,
                training_seed=result.training_seed,
                git_commit_hash=result.git_commit_hash,
                metrics_hash=result.metrics_hash,
                artifact_hash=result.artifact_hash,
            )
            register_training_run(
                session,
                experiment_id=None,
                model_version=result.version,
                status="completed",
                metrics=result.metrics,
                hyperparameters={
                    "model_type": training_config.model_type,
                    "hyperparameters": training_config.hyperparameters,
                    "search": {
                        "method": training_config.search.method,
                        "param_grid": training_config.search.param_grid,
                        "n_iter": training_config.search.n_iter,
                        "cv_folds": training_config.search.cv_folds,
                    },
                },
                dataset_dir=str(training_config.dataset_dir),
                feature_schema_version=result.feature_schema_version,
                artifact_path=str(result.artifact_dir),
                dataset_snapshot_id=result.dataset_snapshot_id,
            )

        run_with_db_retry(_register, commit=True, operation_name="register_retrain")
        model_version = result.version
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)

    def _mark_done(session):
        job = _get_job(session, job_id)
        mark_job_completed(
            session, job, model_version=model_version, error_message=error_message
        )

    run_with_db_retry(_mark_done, commit=True, operation_name="mark_retrain_complete")


def _get_job(session, job_id):
    from app.db.models import RetrainJob

    job = session.get(RetrainJob, job_id)
    if job is None:
        raise ValueError("Retrain job not found")
    return job
