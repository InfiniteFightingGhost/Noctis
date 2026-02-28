from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import get_settings
from app.db.models import Epoch, ModelVersion, Prediction
from app.evaluation.stats import stage_distribution, transition_matrix
from app.feature_store.service import get_feature_schema_by_version
from app.ml.feature_decode import decode_features
from app.ml.feature_schema import load_feature_schema
from app.ml.model import load_model
from app.ml.validation import ensure_finite, prepare_batch
from app.services.windowing import WindowedEpoch, build_windows


def replay_models(
    session: Session,
    *,
    tenant_id,
    recording_id: str,
    model_version_a: str,
    model_version_b: str,
) -> dict[str, Any]:
    settings = get_settings()
    model_a, schema_a, metadata_a = _load_version(session, model_version_a)
    model_b, schema_b, metadata_b = _load_version(session, model_version_b)
    if schema_a.version != schema_b.version:
        raise ValueError("Model feature schema mismatch")
    epochs = _fetch_epochs(session, tenant_id, recording_id, schema_a)
    if not epochs:
        raise ValueError("No epochs found for recording")
    windows, window_end_ts = _build_windows(
        epochs, settings.window_size, epoch_seconds=settings.epoch_seconds
    )
    preds_a, conf_a = _predict(model_a, windows, metadata_a, schema_a.size)
    preds_b, conf_b = _predict(model_b, windows, metadata_b, schema_b.size)
    ground_truth = _load_ground_truth(session, tenant_id, recording_id, window_end_ts)
    summary = _build_summary(preds_a, preds_b, conf_a, conf_b, ground_truth)
    return {
        "recording_id": recording_id,
        "model_version_a": model_version_a,
        "model_version_b": model_version_b,
        "generated_at": datetime.now(timezone.utc),
        "summary": summary,
    }


def _fetch_epochs(
    session: Session,
    tenant_id,
    recording_id: str,
    feature_schema: Any,
) -> list[WindowedEpoch]:
    rows = session.execute(
        select(
            Epoch.epoch_index,
            Epoch.epoch_start_ts,
            Epoch.feature_schema_version,
            Epoch.features_payload,
        )
        .where(Epoch.recording_id == recording_id)
        .where(Epoch.tenant_id == tenant_id)
        .order_by(Epoch.epoch_index)
    ).all()
    if not rows:
        return []
    schema_version = rows[0][2]
    if schema_version != feature_schema.version:
        raise ValueError("Feature schema mismatch for replay")
    schema_id = getattr(feature_schema, "schema_id", None)
    if schema_id is None:
        record = get_feature_schema_by_version(session, feature_schema.version)
        if record is None:
            raise ValueError("Feature schema not registered")
        schema_id = record.id
    epochs: list[WindowedEpoch] = []
    for epoch_index, epoch_start_ts, _, payload in rows:
        payload_features = payload.get("features") if isinstance(payload, dict) else None
        if payload_features is None:
            raise ValueError("Missing features payload for replay")
        vector = decode_features(payload_features, feature_schema)
        ensure_finite("features", vector)
        epochs.append(
            WindowedEpoch(
                epoch_index=epoch_index,
                epoch_start_ts=epoch_start_ts,
                features=vector,
                feature_schema_id=schema_id,
            )
        )
    return epochs


def _build_windows(
    epochs: list[WindowedEpoch],
    window_size: int,
    *,
    epoch_seconds: int,
) -> tuple[list[np.ndarray], list[datetime]]:
    windows = build_windows(
        epochs,
        window_size=window_size,
        allow_padding=False,
        epoch_seconds=epoch_seconds,
    )
    tensors = [window.tensor for window in windows]
    end_ts = [window.end_ts for window in windows]
    return tensors, end_ts


def _load_version(session: Session, version: str) -> tuple[Any, Any, dict[str, Any]]:
    model = session.execute(
        select(ModelVersion).where(ModelVersion.version == version)
    ).scalar_one_or_none()
    if model is None:
        root = get_settings().model_registry_path
        model_dir = root / version
        bundle = load_model(model_dir)
        schema = load_feature_schema(model_dir / "feature_schema.json")
        return bundle.model, schema, bundle.metadata
    if not model.artifact_path:
        raise ValueError("Model artifact path missing")
    model_dir = Path(model.artifact_path)
    bundle = load_model(model_dir)
    schema = load_feature_schema(model_dir / "feature_schema.json")
    return bundle.model, schema, bundle.metadata


def _predict(
    model: Any,
    windows: list[np.ndarray],
    metadata: dict[str, Any],
    feature_dim: int,
) -> tuple[list[str], list[float]]:
    if not windows:
        return [], []
    feature_strategy = metadata.get("feature_strategy")
    expected_input_dim = metadata.get("expected_input_dim")
    window_size = metadata.get("window_size")
    if feature_strategy is None or expected_input_dim is None or window_size is None:
        raise ValueError("Model metadata missing feature strategy")
    batch = prepare_batch(
        windows,
        feature_strategy=str(feature_strategy),
        expected_input_dim=int(expected_input_dim),
        feature_dim=feature_dim,
        window_size=int(window_size),
    )
    if str(feature_strategy) == "sequence":
        dataset_id = str(metadata.get("inference_dataset_id", "UNKNOWN"))
        dataset_ids = np.full(batch.shape[0], dataset_id, dtype=object)
        probs = model.predict_proba(batch, dataset_ids=dataset_ids)
    else:
        probs = model.predict_proba(batch)
    preds: list[str] = []
    conf: list[float] = []
    for row in probs:
        idx = int(np.argmax(row))
        preds.append(model.labels[idx])
        conf.append(float(row[idx]))
    return preds, conf


def _load_ground_truth(
    session: Session,
    tenant_id,
    recording_id: str,
    window_end_ts: list[datetime],
) -> list[str | None] | None:
    if not window_end_ts:
        return None
    raw_rows = session.execute(
        select(Prediction.window_end_ts, Prediction.ground_truth_stage)
        .where(Prediction.recording_id == recording_id)
        .where(Prediction.tenant_id == tenant_id)
        .where(Prediction.window_end_ts.in_(window_end_ts))
    ).all()
    rows: list[tuple[datetime, str | None]] = [(row[0], row[1]) for row in raw_rows]
    if not rows:
        return None
    lookup: dict[datetime, str] = {row[0]: row[1] for row in rows if row[1]}
    if not lookup:
        return None
    return [lookup.get(ts) for ts in window_end_ts]


def _build_summary(
    preds_a: list[str],
    preds_b: list[str],
    conf_a: list[float],
    conf_b: list[float],
    ground_truth: list[str | None] | None,
) -> dict[str, Any]:
    dist_a = stage_distribution(preds_a)
    dist_b = stage_distribution(preds_b)
    transition_a = transition_matrix(preds_a, sorted(set(preds_a)))
    transition_b = transition_matrix(preds_b, sorted(set(preds_b)))
    transition_delta = _matrix_delta(transition_a, transition_b)
    summary = {
        "stage_distribution": {
            "model_a": dist_a,
            "model_b": dist_b,
        },
        "confidence_delta": {
            "mean": float(np.mean(conf_a) - np.mean(conf_b)) if conf_a and conf_b else 0.0,
            "model_a": float(np.mean(conf_a)) if conf_a else 0.0,
            "model_b": float(np.mean(conf_b)) if conf_b else 0.0,
        },
        "transition_matrix_diff": transition_delta,
    }
    if ground_truth:
        labels = sorted(
            set([label for label in ground_truth if label]) | set(preds_a) | set(preds_b)
        )
        summary["confusion_matrix_diff"] = {
            "labels": labels,
            "model_a": _confusion_matrix(ground_truth, preds_a, labels),
            "model_b": _confusion_matrix(ground_truth, preds_b, labels),
        }
    return summary


def _confusion_matrix(
    y_true: list[str | None],
    y_pred: list[str],
    labels: list[str],
) -> list[list[int]]:
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, pred in zip(y_true, y_pred, strict=False):
        if truth is None:
            continue
        matrix[index[truth]][index[pred]] += 1
    return matrix


def _matrix_delta(a: list[list[int]], b: list[list[int]]) -> dict[str, Any]:
    if not a or not b:
        return {"l1": 0.0, "l2": 0.0}
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    diff = arr_a - arr_b
    return {
        "l1": float(np.sum(np.abs(diff))),
        "l2": float(np.sqrt(np.sum(diff**2))),
    }
