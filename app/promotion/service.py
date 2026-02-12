from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import get_settings
from app.db.models import ModelPromotionEvent, ModelVersion


def promote_model(
    session: Session,
    *,
    version: str,
    actor: str,
    reason: str | None,
) -> ModelVersion:
    model = _get_model(session, version)
    _assert_metrics_thresholds(model)
    current_prod = _get_current_production(session)
    if current_prod and current_prod.version == model.version:
        return model
    if current_prod:
        previous_status = current_prod.status
        _update_status(session, current_prod, "archived", actor, reason)
        _record_event(
            session, current_prod.version, previous_status, "archived", actor, reason
        )
    previous_status = model.status
    _update_status(session, model, "production", actor, reason)
    _record_event(session, model.version, previous_status, "production", actor, reason)
    _activate_model_artifacts(model)
    return model


def archive_model(
    session: Session,
    *,
    version: str,
    actor: str,
    reason: str | None,
) -> ModelVersion:
    model = _get_model(session, version)
    previous_status = model.status
    _update_status(session, model, "archived", actor, reason)
    _record_event(session, model.version, previous_status, "archived", actor, reason)
    return model


def rollback_model(
    session: Session,
    *,
    version: str,
    actor: str,
    reason: str | None,
) -> ModelVersion:
    model = _get_model(session, version)
    current_prod = _get_current_production(session)
    if current_prod and current_prod.version == model.version:
        return model
    if current_prod:
        previous_status = current_prod.status
        _update_status(session, current_prod, "archived", actor, reason)
        _record_event(
            session, current_prod.version, previous_status, "archived", actor, reason
        )
    previous_status = model.status
    _update_status(session, model, "production", actor, reason)
    _record_event(session, model.version, previous_status, "production", actor, reason)
    _activate_model_artifacts(model)
    return model


def _get_model(session: Session, version: str) -> ModelVersion:
    model = session.execute(
        select(ModelVersion).where(ModelVersion.version == version)
    ).scalar_one_or_none()
    if model is None:
        raise ValueError("Model version not found")
    return model


def _get_current_production(session: Session) -> ModelVersion | None:
    return session.execute(
        select(ModelVersion).where(ModelVersion.status == "production")
    ).scalar_one_or_none()


def _assert_metrics_thresholds(model: ModelVersion) -> None:
    settings = get_settings()
    if settings.promotion_block_if_missing_metrics and not model.metrics:
        raise ValueError("Model metrics missing")
    metrics = model.metrics or {}
    min_accuracy = settings.promotion_min_accuracy
    min_f1 = settings.promotion_min_macro_f1
    accuracy = float(metrics.get("accuracy") or 0.0)
    macro_f1 = float(metrics.get("macro_f1") or 0.0)
    if accuracy < min_accuracy:
        raise ValueError("Model accuracy below threshold")
    if macro_f1 < min_f1:
        raise ValueError("Model macro F1 below threshold")


def _update_status(
    session: Session,
    model: ModelVersion,
    status: str,
    actor: str,
    reason: str | None,
) -> None:
    model.status = status
    if status == "production":
        model.promoted_at = datetime.now(timezone.utc)
        model.promoted_by = actor
    if status == "archived":
        model.archived_at = datetime.now(timezone.utc)
    model.details = {
        **(model.details or {}),
        "last_transition": {
            "to": status,
            "actor": actor,
            "reason": reason,
            "at": datetime.now(timezone.utc).isoformat(),
        },
    }
    session.add(model)


def _record_event(
    session: Session,
    version: str,
    previous_status: str,
    new_status: str,
    actor: str,
    reason: str | None,
) -> None:
    session.add(
        ModelPromotionEvent(
            model_version=version,
            previous_status=previous_status,
            new_status=new_status,
            actor=actor,
            reason=reason,
        )
    )


def _activate_model_artifacts(model: ModelVersion) -> None:
    settings = get_settings()
    root = settings.model_registry_path
    root.mkdir(parents=True, exist_ok=True)
    source = Path(model.artifact_path)
    if not source.exists():
        raise ValueError("Model artifacts not found")
    temp_base = Path(tempfile.mkdtemp(prefix="model_active_", dir=root))
    temp_dir = temp_base / "active"
    shutil.copytree(source, temp_dir)
    active_dir = root / settings.active_model_version
    if active_dir.exists():
        shutil.rmtree(active_dir)
    temp_dir.replace(active_dir)
    if temp_base.exists():
        shutil.rmtree(temp_base)
