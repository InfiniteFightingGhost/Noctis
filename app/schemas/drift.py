from __future__ import annotations

from datetime import datetime

from app.schemas.common import BaseSchema


class DriftMetric(BaseSchema):
    name: str
    psi: float | None
    kl_divergence: float | None
    z_score: float | None
    current_window: int
    baseline_window: int
    status: str
    severity: str
    drift_score: float


class FeatureDrift(BaseSchema):
    feature_index: int
    current_mean: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    current_window: int
    baseline_window: int
    status: str
    severity: str
    drift_score: float


class FeatureFlag(BaseSchema):
    feature_index: int
    severity: str
    z_score: float | None = None


class DriftResponse(BaseSchema):
    model_version: str | None
    from_ts: datetime | None
    to_ts: datetime | None
    window_size: int
    thresholds: dict[str, float]
    metrics: list[DriftMetric]
    feature_drift: list[FeatureDrift] = []
    flagged_features: list[FeatureFlag] = []
    overall_severity: str
    generated_at: datetime
