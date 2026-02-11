from __future__ import annotations

from collections import Counter
from datetime import datetime

from app.schemas.summary import RecordingSummary


STAGE_MINUTES = 0.5


def summarize_predictions(
    recording_id: str,
    from_ts: datetime,
    to_ts: datetime,
    stages: list[str],
) -> RecordingSummary:
    if not stages:
        return RecordingSummary(
            recording_id=recording_id,
            from_ts=from_ts,
            to_ts=to_ts,
            total_minutes=0.0,
            time_in_stage_minutes={},
            sleep_latency_minutes=None,
            waso_minutes=None,
        )
    counts = Counter(stages)
    time_in_stage = {stage: count * STAGE_MINUTES for stage, count in counts.items()}
    total_minutes = len(stages) * STAGE_MINUTES

    sleep_latency_minutes = None
    waso_minutes = None
    try:
        first_sleep_idx = next(i for i, stage in enumerate(stages) if stage != "W")
        sleep_latency_minutes = first_sleep_idx * STAGE_MINUTES
        sleep_onset = stages[first_sleep_idx:]
        waso_minutes = sum(1 for stage in sleep_onset if stage == "W") * STAGE_MINUTES
    except StopIteration:
        sleep_latency_minutes = None
        waso_minutes = None

    return RecordingSummary(
        recording_id=recording_id,
        from_ts=from_ts,
        to_ts=to_ts,
        total_minutes=total_minutes,
        time_in_stage_minutes=time_in_stage,
        sleep_latency_minutes=sleep_latency_minutes,
        waso_minutes=waso_minutes,
    )
