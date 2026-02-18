from __future__ import annotations

import argparse
from pathlib import Path

from app.core.settings import get_settings
from app.db.session import run_with_db_retry
from app.experiments.service import (
    create_experiment_if_missing,
    register_model_version,
    register_training_run,
)
from app.training.config import load_training_config
from app.training.trainer import train_model
from app.training.versioning import next_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model offline")
    parser.add_argument("--config", required=True, help="Path to training config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(Path(args.config))

    def _get_version(session):
        return next_version(
            session=session, output_root=config.output_root, bump=config.version_bump
        )

    version = run_with_db_retry(_get_version, operation_name="next_model_version")
    result = train_model(config=config, version=version)

    def _register(session):
        experiment_id = None
        if config.experiment_name:
            experiment = create_experiment_if_missing(
                session,
                config.experiment_name,
                tenant_id=get_settings().default_tenant_id,
            )
            experiment_id = experiment.id
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
            experiment_id=experiment_id,
            model_version=result.version,
            status="completed",
            metrics=result.metrics,
            hyperparameters={
                "model_type": config.model_type,
                "hyperparameters": config.hyperparameters,
                "search": {
                    "method": config.search.method,
                    "param_grid": config.search.param_grid,
                    "n_iter": config.search.n_iter,
                    "cv_folds": config.search.cv_folds,
                },
            },
            dataset_dir=str(config.dataset_dir),
            feature_schema_version=result.feature_schema_version,
            artifact_path=str(result.artifact_dir),
            dataset_snapshot_id=result.dataset_snapshot_id,
        )

    run_with_db_retry(_register, commit=True, operation_name="register_training_run")


if __name__ == "__main__":
    main()
