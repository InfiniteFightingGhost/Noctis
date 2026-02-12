from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query

from app.auth.dependencies import require_scopes
from app.core.metrics import DRIFT_REQUESTS, DRIFT_SCORE_GAUGE, DRIFT_SEVERITY_GAUGE
from app.core.settings import get_settings
from app.db.session import run_with_db_retry
from app.drift.service import compute_drift
from app.scheduler.service import enqueue_retrain_job
from app.schemas.drift import DriftResponse
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["drift"], dependencies=[Depends(require_scopes("read"))])


@router.get("/model/drift", response_model=DriftResponse)
def model_drift(
    model_version: str | None = None,
    from_ts: datetime | None = Query(default=None, alias="from"),
    to_ts: datetime | None = Query(default=None, alias="to"),
    tenant: TenantContext = Depends(get_tenant_context),
) -> dict:
    settings = get_settings()

    def _op(session):
        return compute_drift(
            session,
            tenant_id=tenant.id,
            model_version=model_version,
            from_ts=from_ts,
            to_ts=to_ts,
            window_size=settings.performance_sample_size,
        )

    DRIFT_REQUESTS.inc()
    payload = run_with_db_retry(_op, operation_name="model_drift")
    thresholds = {
        "psi": settings.drift_psi_threshold,
        "kl": settings.drift_kl_threshold,
        "z": settings.drift_z_threshold,
    }
    flagged_features = []
    alert_count = 0
    for metric in payload["metrics"]:
        status, severity, score = _classify_metric(metric, thresholds)
        metric["status"] = status
        metric["severity"] = severity
        metric["drift_score"] = score
        DRIFT_SCORE_GAUGE.labels(metric=metric["name"]).set(score)
        if status == "alert":
            alert_count += 1

    for feature in payload.get("feature_drift", []):
        status, severity, score = _classify_feature(feature, thresholds)
        feature["status"] = status
        feature["severity"] = severity
        feature["drift_score"] = score
        DRIFT_SCORE_GAUGE.labels(metric=f"feature_{feature['feature_index']}").set(
            score
        )
        if status == "alert":
            alert_count += 1
            flagged_features.append(
                {
                    "feature_index": feature["feature_index"],
                    "severity": severity,
                    "z_score": feature.get("z_score"),
                }
            )
    overall_severity = _overall_severity(
        payload["metrics"], payload.get("feature_drift", []), alert_count
    )
    DRIFT_SEVERITY_GAUGE.labels(scope="model").set(_severity_value(overall_severity))
    max_drift_score = max(
        [metric.get("drift_score", 0.0) for metric in payload["metrics"]]
        + [
            feature.get("drift_score", 0.0)
            for feature in payload.get("feature_drift", [])
        ]
        or [0.0]
    )
    payload.update(
        {
            "model_version": model_version,
            "thresholds": thresholds,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "flagged_features": flagged_features,
            "overall_severity": overall_severity,
        }
    )
    if max_drift_score >= settings.retrain_drift_threshold and alert_count > 0:
        _schedule_retrain(
            payload,
            flagged_features,
            settings,
            tenant.id,
            from_ts,
            to_ts,
        )
    return payload


def _schedule_retrain(
    payload: dict,
    flagged_features: list[dict],
    settings,
    tenant_id,
    from_ts: datetime | None,
    to_ts: datetime | None,
) -> None:
    suggested_from = from_ts or datetime.now(timezone.utc) - timedelta(days=30)
    suggested_to = to_ts or datetime.now(timezone.utc)
    output_dir = (
        settings.retrain_dataset_output_root
        / f"retrain_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    )
    dataset_config = {
        "output_dir": str(output_dir),
        "feature_schema_path": str(settings.retrain_feature_schema_path),
        "window_size": settings.window_size,
        "allow_padding": settings.allow_window_padding,
        "label_strategy": "ground_truth_or_predicted",
        "balance_strategy": "none",
        "random_seed": 42,
        "export_format": "npz",
        "split": {"train": 0.7, "val": 0.15, "test": 0.15},
        "filters": {
            "from_ts": suggested_from.isoformat(),
            "to_ts": suggested_to.isoformat(),
            "feature_schema_version": settings.feature_schema_version,
            "tenant_id": str(tenant_id),
        },
    }
    training_config = {
        "dataset_dir": str(output_dir),
        "output_root": str(settings.retrain_model_output_root),
        "feature_schema_path": str(settings.retrain_feature_schema_path),
        "model_type": "gradient_boosting",
        "random_seed": 42,
        "class_balance": "none",
        "feature_strategy": "mean",
        "hyperparameters": {},
        "search": {"method": "random", "param_grid": {}, "n_iter": 5, "cv_folds": 3},
        "version_bump": "patch",
        "experiment_name": settings.retrain_experiment_name,
    }

    def _op(session):
        enqueue_retrain_job(
            session,
            tenant_id=tenant_id,
            drift_score=max(
                [metric.get("drift_score", 0.0) for metric in payload["metrics"]]
                + [
                    feature.get("drift_score", 0.0)
                    for feature in payload.get("feature_drift", [])
                ]
                or [0.0]
            ),
            triggering_features=flagged_features,
            suggested_from_ts=suggested_from,
            suggested_to_ts=suggested_to,
            dataset_config=dataset_config,
            training_config=training_config,
        )

    run_with_db_retry(_op, commit=True, operation_name="enqueue_retrain")


def _classify_metric(
    metric: dict, thresholds: dict[str, float]
) -> tuple[str, str, float]:
    score, threshold = _metric_score_threshold(metric, thresholds)
    status = "alert" if score >= threshold else "ok"
    severity = _severity(score, threshold)
    return status, severity, score


def _classify_feature(
    metric: dict, thresholds: dict[str, float]
) -> tuple[str, str, float]:
    score = abs(float(metric.get("z_score") or 0.0))
    threshold = thresholds["z"]
    status = "alert" if score >= threshold else "ok"
    severity = _severity(score, threshold)
    return status, severity, score


def _metric_score_threshold(
    metric: dict, thresholds: dict[str, float]
) -> tuple[float, float]:
    psi_value = metric.get("psi")
    kl_value = metric.get("kl_divergence")
    if psi_value is not None and kl_value is not None:
        if float(psi_value) >= float(kl_value):
            return float(psi_value), thresholds["psi"]
        return float(kl_value), thresholds["kl"]
    if psi_value is not None:
        return float(psi_value), thresholds["psi"]
    if kl_value is not None:
        return float(kl_value), thresholds["kl"]
    z_value = metric.get("z_score")
    return (abs(float(z_value)) if z_value is not None else 0.0), thresholds["z"]


def _severity(score: float, threshold: float) -> str:
    if threshold <= 0:
        return "LOW"
    if score >= threshold * 2:
        return "HIGH"
    if score >= threshold:
        return "MEDIUM"
    return "LOW"


def _overall_severity(
    metrics: list[dict], feature_drift: list[dict], alert_count: int
) -> str:
    severity_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    all_severities = [
        item.get("severity")
        for item in [*metrics, *feature_drift]
        if item.get("severity")
    ]
    max_severity = max(
        all_severities,
        default="LOW",
        key=lambda value: severity_order.get(value, 0),
    )
    if alert_count > 0 and severity_order.get(max_severity, 0) < 1:
        return "MEDIUM"
    return max_severity


def _severity_value(level: str) -> int:
    return {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(level, 0)
