from __future__ import annotations

from datetime import datetime

from app.schemas.common import BaseSchema


class StagePercentages(BaseSchema):
    awake: int
    light: int
    deep: int
    rem: int


class StageBin(BaseSchema):
    startMinFromBedtime: int
    durationMin: int
    stage: str


class SleepTotals(BaseSchema):
    totalSleepMin: int
    timeInBedMin: int
    sleepEfficiencyPct: int


class SleepMetrics(BaseSchema):
    deepPct: int
    avgHrBpm: int
    avgRrBrpm: int
    movementPct: int


class SleepInsight(BaseSchema):
    text: str
    tag: str | None = None
    confidence: float | None = None


class PrimaryAction(BaseSchema):
    label: str
    action: str


class DataQuality(BaseSchema):
    status: str
    issues: list[str] = []
    lastSyncAtLocal: str | None = None


class SleepSummaryResponse(BaseSchema):
    recordingId: str
    dateLocal: str
    bedtimeLocal: str
    waketimeLocal: str
    score: int
    scoreLabel: str
    totals: SleepTotals
    stages: dict
    metrics: SleepMetrics
    insight: SleepInsight
    primaryAction: PrimaryAction
    dataQuality: DataQuality


class SyncStatusResponse(BaseSchema):
    status: str
    lastSyncAtLocal: str | None = None


class HomeOverviewResponse(BaseSchema):
    headline: str
    lede: str
    updated_at: str | None = None


class InsightFeedbackRequest(BaseSchema):
    recordingId: str
    feedback: str


class CoachSummaryResponse(BaseSchema):
    generated_at: datetime
    is_partial: bool
    insights: list[dict]
