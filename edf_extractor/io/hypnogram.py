from __future__ import annotations

import csv
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

ISRUC_STAGE_TO_CODE = {
    0: STAGE_WAKE,
    1: STAGE_N1,
    2: STAGE_N2,
    3: STAGE_N3,
    5: STAGE_REM,
}


def read_hypnogram(
    path: str | Path,
    epoch_sec: int,
) -> tuple[np.ndarray, str | None, list[str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"Empty hypnogram file in {path}")

    if _looks_like_isruc(lines):
        return read_isruc_hypnogram(lines)
    return read_cap_hypnogram(path, epoch_sec)


def merge_hypnogram_tracks(
    cap_hypnogram: np.ndarray,
    epoch_sec: int,
    annotations: list[tuple[float, float, str]],
) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    if not annotations:
        return cap_hypnogram, warnings
    merged = cap_hypnogram.copy()
    for onset_sec, duration_sec, label in sorted(annotations, key=lambda item: (item[0], item[2])):
        stage = _annotation_stage_to_code(label)
        if stage is None:
            continue
        start_idx = max(0, int(round(onset_sec / epoch_sec)))
        count = max(1, int(round(max(duration_sec, epoch_sec) / epoch_sec)))
        for idx in range(start_idx, min(merged.shape[0], start_idx + count)):
            if merged[idx] != STAGE_UNKNOWN and merged[idx] != stage:
                warnings.append("annotation_cap_conflict")
                continue
            merged[idx] = stage
    return merged, sorted(set(warnings))


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
            parts = _split_line(line)
            row = _extract_stage_row(parts)
            if row is None:
                continue
            event, time_text, duration_text = row
            start_sec, normalized_time = _parse_time(time_text)
            if previous_sec is not None and start_sec < previous_sec:
                offset_sec += 24 * 3600
            absolute_sec = start_sec + offset_sec
            previous_sec = start_sec
            if first_clock is None:
                first_clock = normalized_time

            duration_sec = int(round(_parse_duration_seconds(duration_text)))
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


def read_isruc_hypnogram(lines: list[str]) -> tuple[np.ndarray, str | None, list[str]]:
    warnings: list[str] = []
    values: list[int] = []
    for idx, line in enumerate(lines):
        token = line.split()[0]
        try:
            raw_stage = int(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid ISRUC stage value at line {idx + 1}: {line}") from exc
        mapped = ISRUC_STAGE_TO_CODE.get(raw_stage, STAGE_UNKNOWN)
        if mapped == STAGE_UNKNOWN and raw_stage not in ISRUC_STAGE_TO_CODE:
            warnings.append("unknown_isruc_stage_code")
        values.append(mapped)
    return np.asarray(values, dtype=np.int8), None, warnings


def _looks_like_isruc(lines: list[str]) -> bool:
    probe = lines[: min(20, len(lines))]
    for line in probe:
        parts = line.split()
        if len(parts) != 1:
            return False
        try:
            float(parts[0])
        except ValueError:
            return False
    return True


def _parse_time(value: str) -> tuple[int, str]:
    match = re.search(r"(\d{1,2})[:.](\d{2})[:.](\d{2})", value)
    if match is None:
        raise ValueError(f"Unsupported time format: {value}")
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    normalized = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return hours * 3600 + minutes * 60 + seconds, normalized


def _split_line(line: str) -> list[str]:
    if "\t" in line:
        return [part.strip() for part in line.split("\t")]
    if "," in line:
        return _parse_csv_line(line, ",")
    if ";" in line:
        return _parse_csv_line(line, ";")
    return line.split()


def _parse_csv_line(line: str, delimiter: str) -> list[str]:
    row = next(csv.reader([line], delimiter=delimiter), [])
    return [part.strip() for part in row]


def _annotation_stage_to_code(label: str) -> int | None:
    normalized = label.strip().upper()
    if not normalized:
        return None
    if "REM" in normalized:
        return STAGE_REM
    if "WAKE" in normalized or normalized.endswith("W"):
        return STAGE_WAKE
    if "N1" in normalized or "S1" in normalized:
        return STAGE_N1
    if "N2" in normalized or "S2" in normalized:
        return STAGE_N2
    if "N3" in normalized or "S3" in normalized or "S4" in normalized:
        return STAGE_N3
    if "UNSCORED" in normalized:
        return STAGE_UNKNOWN
    return None


def _extract_stage_row(parts: list[str]) -> tuple[str, str, str] | None:
    if len(parts) < 3:
        return None
    event_idx = None
    event_name = ""
    for idx, token in enumerate(parts):
        candidate = token.strip().upper()
        if candidate in STAGE_EVENT_TO_CODE:
            event_idx = idx
            event_name = candidate
            break
    if event_idx is None:
        return None

    time_text = ""
    for token in parts:
        value = token.strip()
        if re.search(r"\d{1,2}[:.]\d{2}[:.]\d{2}", value):
            time_text = value
            break
    if not time_text:
        return None

    duration_text = ""
    for idx in range(event_idx + 1, len(parts)):
        token = parts[idx].strip()
        if _is_number(token):
            duration_text = token
            break
    if not duration_text:
        return None

    return event_name, time_text, duration_text


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _parse_duration_seconds(value: str) -> float:
    cleaned = value.strip().replace(",", ".")
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Unsupported duration format: {value}") from exc
