from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from edf_extractor.constants import (
    STAGE_N1,
    STAGE_N2,
    STAGE_N3,
    STAGE_REM,
    STAGE_UNKNOWN,
    STAGE_WAKE,
)

STAGE_EVENT_TO_CODE = {
    "SLEEP-S0": STAGE_WAKE,
    "SLEEP-S1": STAGE_N1,
    "SLEEP-S2": STAGE_N2,
    "SLEEP-S3": STAGE_N3,
    "SLEEP-S4": STAGE_N3,
    "SLEEP-REM": STAGE_REM,
    "SLEEP-UNSCORED": STAGE_UNKNOWN,
    "SLEEP-MT": STAGE_UNKNOWN,
}


def read_cap_hypnogram(
    path: str | Path, epoch_sec: int
) -> tuple[np.ndarray, str | None, list[str]]:
    path = Path(path)
    warnings: list[str] = []
    events: list[tuple[int, int, int]] = []
    first_clock: str | None = None
    offset_sec = 0
    previous_sec: int | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            event = parts[3].strip().upper()
            if event not in STAGE_EVENT_TO_CODE:
                continue
            time_text = parts[2].strip()
            duration_text = parts[4].strip()
            start_sec, normalized_time = _parse_time(time_text)
            if previous_sec is not None and start_sec < previous_sec:
                offset_sec += 24 * 3600
            absolute_sec = start_sec + offset_sec
            previous_sec = start_sec
            if first_clock is None:
                first_clock = normalized_time

            duration_sec = int(round(float(duration_text)))
            if duration_sec <= 0:
                continue
            if duration_sec % epoch_sec != 0:
                warnings.append("non_30s_stage_duration")
            epoch_count = max(1, int(round(duration_sec / epoch_sec)))
            events.append((absolute_sec, epoch_count, STAGE_EVENT_TO_CODE[event]))

    if not events:
        raise ValueError(f"No stage events found in {path}")

    base_sec = events[0][0]
    last_idx = 0
    for start_sec, count, _ in events:
        idx = int(round((start_sec - base_sec) / epoch_sec))
        last_idx = max(last_idx, idx + count)

    hypnogram = np.full(last_idx, STAGE_UNKNOWN, dtype=np.int8)
    for start_sec, count, stage in events:
        idx = int(round((start_sec - base_sec) / epoch_sec))
        end_idx = min(last_idx, idx + count)
        hypnogram[idx:end_idx] = stage

    return hypnogram, first_clock, warnings


def _parse_time(value: str) -> tuple[int, str]:
    match = re.search(r"(\d{1,2})[:.](\d{2})[:.](\d{2})", value)
    if match is None:
        raise ValueError(f"Unsupported time format: {value}")
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    normalized = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return hours * 3600 + minutes * 60 + seconds, normalized
