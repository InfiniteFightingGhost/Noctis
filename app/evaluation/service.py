from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Any, cast
import json
from pathlib import Path
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import ModelUsageStat, Prediction, ModelVersion
from app.drift.stats import mean, std, z_score
from app.evaluation.stats import (
    accuracy,
    average_confidence,
    build_labels,
    confidence_histogram,
    confusion_matrix,
    entropy_metrics,
    merge_labels,
    macro_f1,
    night_summary_delta,
    night_summary_metrics,
    per_class_frequency,
    per_class_metrics,
    prediction_distribution,
    transition_matrix,
)


def _apply_prediction_filters(
    query: Any,
    *,
    tenant_id: uuid.UUID,
    recording_id: uuid.UUID | None = None,
    model_version: str | None = None,
    from_ts: datetime | None = None,
    to_ts: datetime | None = None,
):
    query = query.filter(Prediction.tenant_id == tenant_id)
    if recording_id:
        query = query.filter(Prediction.recording_id == recording_id)
    if model_version:
        query = query.filter(Prediction.model_version == model_version)
    if from_ts:
        query = query.filter(Prediction.window_end_ts >= from_ts)
    if to_ts:
        query = query.filter(Prediction.window_end_ts <= to_ts)
    return query


def compute_evaluation(
    session: Session,
    *,
    tenant_id: uuid.UUID,
    recording_id: uuid.UUID | None = None,
    model_version: str | None = None,
    from_ts: datetime | None = None,
    to_ts: datetime | None = None,
) -> dict[str, Any]:
    query = session.query(
        Prediction.predicted_stage,
        Prediction.ground_truth_stage,
        Prediction.confidence,
        Prediction.probabilities,
        Prediction.window_end_ts,
    )
    query = _apply_prediction_filters(
        query,
        tenant_id=tenant_id,
        recording_id=recording_id,
        model_version=model_version,
        from_ts=from_ts,
        to_ts=to_ts,
    )
    query = query.order_by(Prediction.window_end_ts)
    rows = query.all()
    model_label_map = _resolve_label_map(session, model_version)

    predicted: list[str] = []
    truth: list[str] = []
    confidences: list[float] = []
    probabilities: list[dict[str, float]] = []
    sequence: list[str] = []
    for pred, gt, confidence, probs, _ts in rows:
        predicted.append(pred)
        sequence.append(pred)
        confidence_value = float(confidence)
        if not isfinite(confidence_value):
            confidence_value = 0.0
        confidences.append(confidence_value)
        probabilities.append(
            {str(k): (float(v) if isfinite(float(v)) else 0.0) for k, v in probs.items()}
        )
        if gt:
            truth.append(gt)
        else:
            truth.append("__unlabeled__")

    labeled_pairs = [(t, p) for t, p in zip(truth, predicted, strict=True) if t != "__unlabeled__"]
    labeled_truth = [t for t, _ in labeled_pairs]
    labeled_pred = [p for _, p in labeled_pairs]
    if labeled_pairs:
        if model_label_map:
            labels = merge_labels(model_label_map, labeled_truth + labeled_pred)
        else:
            labels = build_labels(labeled_truth, labeled_pred)
    else:
        labels = []
    matrix = confusion_matrix(labeled_truth, labeled_pred, labels) if labels else []
    per_class = per_class_metrics(matrix, labels) if labels else []
    acc = accuracy(matrix) if labels else 0.0
    macro = macro_f1(per_class) if labels else 0.0
    if model_label_map:
        transition_labels = merge_labels(model_label_map, sequence)
    else:
        transition_labels = sorted(set(sequence))
    transitions = transition_matrix(sequence, transition_labels)

    avg_confidence = average_confidence(confidences)
    distribution = prediction_distribution(predicted)
    class_frequency = per_class_frequency(predicted)

    night_validation = None
    if labeled_pairs:
        predicted_summary = night_summary_metrics(labeled_pred)
        truth_summary = night_summary_metrics(labeled_truth)
        night_validation = {
            "predicted": predicted_summary,
            "ground_truth": truth_summary,
            "delta": night_summary_delta(predicted_summary, truth_summary),
        }

    return {
        "total_predictions": len(predicted),
        "labeled_predictions": len(labeled_pairs),
        "accuracy": acc if labeled_pairs else None,
        "macro_f1": macro if labeled_pairs else None,
        "confusion_matrix": {"labels": labels, "matrix": matrix} if labels else None,
        "per_class": per_class,
        "average_confidence": avg_confidence,
        "prediction_distribution": distribution,
        "per_class_frequency": class_frequency,
        "transition_matrix": {"labels": transition_labels, "matrix": transitions},
        "confidence_histogram": confidence_histogram(confidences),
        "entropy": entropy_metrics(probabilities),
        "night_summary_validation": night_validation,
        "generated_at": datetime.now(timezone.utc),
    }


def _resolve_label_map(session: Session, model_version: str | None) -> list[str] | None:
    if not model_version:
        return None
    model = session.execute(
        select(ModelVersion).where(ModelVersion.version == model_version)
    ).scalar_one_or_none()
    if model is None or not model.artifact_path:
        return None
    model_dir = Path(model.artifact_path)
    label_path = model_dir / "label_map.json"
    if label_path.exists():
        label_map = json.loads(label_path.read_text())
        return [str(label) for label in label_map] if isinstance(label_map, list) else None
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        label_map = metadata.get("label_map")
        return [str(label) for label in label_map] if isinstance(label_map, list) else None
    return None


def compute_rolling_evaluation(
    session: Session,
    *,
    tenant_id: uuid.UUID,
    model_version: str | None = None,
    days: int = 7,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    from_ts = now - timedelta(days=days)
    return compute_evaluation(
        session,
        tenant_id=tenant_id,
        model_version=model_version,
        from_ts=from_ts,
        to_ts=now,
    )


def compute_confidence_drift(
    session: Session,
    *,
    tenant_id: uuid.UUID,
    model_version: str | None = None,
    days: int = 7,
) -> dict[str, float]:
    now = datetime.now(timezone.utc)
    from_ts = now - timedelta(days=days * 2)
    query = session.query(Prediction.confidence, Prediction.window_end_ts)
    query = query.filter(Prediction.tenant_id == tenant_id)
    if model_version:
        query = query.filter(Prediction.model_version == model_version)
    query = query.filter(Prediction.window_end_ts >= from_ts)
    rows = query.order_by(Prediction.window_end_ts).all()
    cutoff = now - timedelta(days=days)
    baseline = [float(row[0]) for row in rows if row[1] < cutoff]
    current = [float(row[0]) for row in rows if row[1] >= cutoff]
    baseline_mean = mean(baseline)
    baseline_std = std(baseline)
    current_mean = mean(current)
    return {
        "current_mean": current_mean,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "z_score": z_score(current_mean, baseline_mean, baseline_std),
    }


def compute_model_usage_stats(
    session: Session,
    *,
    tenant_id: uuid.UUID,
    model_version: str | None = None,
    from_ts: datetime | None = None,
    to_ts: datetime | None = None,
) -> list[dict[str, object]]:
    query = session.query(
        ModelUsageStat.model_version,
        func.sum(ModelUsageStat.prediction_count).label("prediction_count"),
        func.min(ModelUsageStat.window_start_ts).label("window_start_ts"),
        func.max(ModelUsageStat.window_end_ts).label("window_end_ts"),
        func.sum(ModelUsageStat.prediction_count * ModelUsageStat.average_latency_ms).label(
            "weighted_latency"
        ),
    )
    query = query.filter(ModelUsageStat.tenant_id == tenant_id)
    if model_version:
        query = query.filter(ModelUsageStat.model_version == model_version)
    if from_ts:
        query = query.filter(ModelUsageStat.window_end_ts >= from_ts)
    if to_ts:
        query = query.filter(ModelUsageStat.window_end_ts <= to_ts)
    query = query.group_by(ModelUsageStat.model_version)

    results: list[dict[str, object]] = []
    for row in query.all():
        prediction_count = int(row.prediction_count or 0)
        avg_latency_ms = float(row.weighted_latency) / prediction_count if prediction_count else 0.0
        results.append(
            {
                "model_version": row.model_version,
                "prediction_count": prediction_count,
                "average_latency_ms": avg_latency_ms,
                "window_start_ts": row.window_start_ts,
                "window_end_ts": row.window_end_ts,
            }
        )
    return sorted(results, key=lambda item: cast(str, item["model_version"]))
