from __future__ import annotations

import uuid
from datetime import datetime

from app.schemas.common import BaseSchema


class ConfusionMatrix(BaseSchema):
    labels: list[str]
    matrix: list[list[int]]


class PerClassMetrics(BaseSchema):
    label: str
    precision: float
    recall: float
    f1: float
    support: int


class TransitionMatrix(BaseSchema):
    labels: list[str]
    matrix: list[list[int]]


class ConfidenceHistogram(BaseSchema):
    bins: list[float]
    counts: list[int]


class PerClassFrequency(BaseSchema):
    label: str
    count: int
    frequency: float


class NightSummaryStats(BaseSchema):
    total_minutes: float
    total_sleep_minutes: float
    sleep_efficiency: float
    sleep_latency_minutes: float | None
    waso_minutes: float | None
    stage_proportions: dict[str, float]


class NightSummaryValidation(BaseSchema):
    predicted: NightSummaryStats
    ground_truth: NightSummaryStats
    delta: dict[str, float | None | dict[str, float]]


class ModelUsageSummary(BaseSchema):
    model_version: str
    prediction_count: int
    average_latency_ms: float
    window_start_ts: datetime | None
    window_end_ts: datetime | None


class ConfidenceDriftSummary(BaseSchema):
    current_mean: float
    baseline_mean: float
    baseline_std: float
    z_score: float


class RollingEvaluation(BaseSchema):
    total_predictions: int
    labeled_predictions: int
    accuracy: float | None
    macro_f1: float | None
    average_confidence: float
    prediction_distribution: dict[str, float]
    per_class_frequency: list[PerClassFrequency]
    entropy: dict[str, float]


class EvaluationResponse(BaseSchema):
    scope: str
    recording_id: uuid.UUID | None
    model_version: str | None
    from_ts: datetime | None = None
    to_ts: datetime | None = None
    total_predictions: int
    labeled_predictions: int
    accuracy: float | None
    macro_f1: float | None
    confusion_matrix: ConfusionMatrix | None
    per_class: list[PerClassMetrics]
    average_confidence: float
    prediction_distribution: dict[str, float]
    per_class_frequency: list[PerClassFrequency]
    transition_matrix: TransitionMatrix
    confidence_histogram: ConfidenceHistogram
    entropy: dict[str, float]
    night_summary_validation: NightSummaryValidation | None = None
    rolling_7_day: RollingEvaluation | None = None
    confidence_drift: ConfidenceDriftSummary | None = None
    confidence_drift_threshold: float | None = None
    model_usage_stats: list[ModelUsageSummary] | None = None
    generated_at: datetime
