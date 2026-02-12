from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import FeatureStatistic, Prediction
from app.drift.stats import kl_divergence, mean, psi, std, z_score
from app.evaluation.stats import stage_distribution
from app.services.feature_stats import extract_means, merge_feature_stats


def _split_windows(items: list[Any], window_size: int) -> tuple[list[Any], list[Any]]:
    if window_size <= 0:
        return [], []
    current = items[-window_size:]
    baseline = items[-(2 * window_size) : -window_size] if len(items) >= 2 else []
    return current, baseline


def compute_drift(
    session: Session,
    *,
    tenant_id,
    model_version: str | None = None,
    from_ts: datetime | None = None,
    to_ts: datetime | None = None,
    window_size: int = 250,
) -> dict[str, Any]:
    query = session.query(
        Prediction.predicted_stage,
        Prediction.confidence,
        Prediction.window_end_ts,
    )
    query = query.filter(Prediction.tenant_id == tenant_id)
    if model_version:
        query = query.filter(Prediction.model_version == model_version)
    if from_ts:
        query = query.filter(Prediction.window_end_ts >= from_ts)
    if to_ts:
        query = query.filter(Prediction.window_end_ts <= to_ts)
    rows = query.order_by(Prediction.window_end_ts.desc()).limit(window_size * 2).all()
    rows = list(reversed(rows))

    stages = [row[0] for row in rows]
    confidences = [float(row[1]) for row in rows]
    current_stages, baseline_stages = _split_windows(stages, window_size)
    current_conf, baseline_conf = _split_windows(confidences, window_size)

    metrics: list[dict[str, Any]] = []
    if current_stages and baseline_stages:
        current_dist = stage_distribution(current_stages)
        baseline_dist = stage_distribution(baseline_stages)
        metrics.append(
            {
                "name": "stage_distribution",
                "psi": psi(current_dist, baseline_dist),
                "kl_divergence": kl_divergence(current_dist, baseline_dist),
                "z_score": None,
                "current_window": len(current_stages),
                "baseline_window": len(baseline_stages),
            }
        )
    if current_conf and baseline_conf:
        metrics.append(
            {
                "name": "confidence_mean",
                "psi": None,
                "kl_divergence": None,
                "z_score": z_score(
                    mean(current_conf), mean(baseline_conf), std(baseline_conf)
                ),
                "current_window": len(current_conf),
                "baseline_window": len(baseline_conf),
            }
        )

    feature_query = session.query(
        FeatureStatistic.stats,
        FeatureStatistic.stat_date,
    )
    feature_query = feature_query.filter(FeatureStatistic.tenant_id == tenant_id)
    if model_version:
        feature_query = feature_query.filter(
            FeatureStatistic.model_version == model_version
        )
    if from_ts:
        feature_query = feature_query.filter(FeatureStatistic.window_end_ts >= from_ts)
    if to_ts:
        feature_query = feature_query.filter(FeatureStatistic.window_end_ts <= to_ts)
    date_query = session.query(FeatureStatistic.stat_date).distinct()
    date_query = date_query.filter(FeatureStatistic.tenant_id == tenant_id)
    if model_version:
        date_query = date_query.filter(FeatureStatistic.model_version == model_version)
    if from_ts:
        date_query = date_query.filter(FeatureStatistic.window_end_ts >= from_ts)
    if to_ts:
        date_query = date_query.filter(FeatureStatistic.window_end_ts <= to_ts)
    stat_dates = [
        row[0]
        for row in date_query.order_by(FeatureStatistic.stat_date.desc())
        .limit(window_size * 2)
        .all()
    ]
    feature_rows = []
    if stat_dates:
        feature_rows = (
            feature_query.filter(FeatureStatistic.stat_date.in_(stat_dates))
            .order_by(FeatureStatistic.stat_date)
            .all()
        )
    feature_drift: list[dict[str, Any]] = []
    if feature_rows:
        aggregated: dict[date, dict] = {}
        for stats, stat_date in feature_rows:
            existing = aggregated.get(stat_date)
            if existing:
                aggregated[stat_date] = merge_feature_stats(existing, stats)
            else:
                aggregated[stat_date] = stats
        ordered_dates = sorted(aggregated.keys())
        means_rows = [
            extract_means(aggregated[stat_date]) for stat_date in ordered_dates
        ]
        current_means, baseline_means = _split_windows(means_rows, window_size)
        if current_means and baseline_means:
            feature_count = len(current_means[0]) if current_means[0] else 0
            for idx in range(feature_count):
                current_values = [row[idx] for row in current_means if len(row) > idx]
                baseline_values = [row[idx] for row in baseline_means if len(row) > idx]
                if not current_values or not baseline_values:
                    continue
                current_mean = mean(current_values)
                baseline_mean = mean(baseline_values)
                baseline_std = std(baseline_values)
                metrics.append(
                    {
                        "name": f"feature_{idx}_mean",
                        "psi": None,
                        "kl_divergence": None,
                        "z_score": z_score(current_mean, baseline_mean, baseline_std),
                        "current_window": len(current_values),
                        "baseline_window": len(baseline_values),
                    }
                )
                feature_drift.append(
                    {
                        "feature_index": idx,
                        "current_mean": current_mean,
                        "baseline_mean": baseline_mean,
                        "baseline_std": baseline_std,
                        "z_score": z_score(current_mean, baseline_mean, baseline_std),
                        "current_window": len(current_values),
                        "baseline_window": len(baseline_values),
                    }
                )

    return {
        "metrics": metrics,
        "feature_drift": feature_drift,
        "generated_at": datetime.now(timezone.utc),
        "window_size": window_size,
    }
