from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from extractor_hardened.errors import ExtractionError

try:
    import pyedflib
except ImportError:  # pragma: no cover - exercised when dependency missing
    pyedflib = None


@dataclass
class EDFSignal:
    label: str
    fs: float
    data: np.ndarray


@dataclass
class EDFRecord:
    signals: list[EDFSignal]
    start_time: str | None
    annotations: list[tuple[float, float, str]]


def read_edf(path: str | Path) -> list[EDFSignal]:
    return read_edf_record(path).signals


def read_edf_record(path: str | Path) -> EDFRecord:
    path = Path(path)
    if path.suffix.lower() == ".rec":
        return read_rec_record(path)

    try:
        return _read_with_pyedflib(path)
    except Exception:
        return _read_edf_record_legacy(path)


def read_rec_record(path: str | Path) -> EDFRecord:
    path = Path(path)
    if pyedflib is None:
        raise ExtractionError(
            code="E_UNSUPPORTED_REC",
            message="REC subtype requires pyedflib for probing",
            details={"path": str(path), "reason": "pyedflib_not_installed"},
        )
    try:
        return _read_with_pyedflib(path)
    except Exception as exc:
        raise ExtractionError(
            code="E_UNSUPPORTED_REC",
            message="REC subtype is not EDF-compatible",
            details={"path": str(path), "reason": type(exc).__name__},
        ) from exc


def _read_with_pyedflib(path: Path) -> EDFRecord:
    if pyedflib is None:
        raise RuntimeError("pyedflib not installed")
    with pyedflib.EdfReader(str(path)) as reader:
        n_signals = int(reader.signals_in_file)
        if n_signals <= 0:
            raise ValueError(f"No EDF signals in {path}")
        labels = [str(label).strip() for label in reader.getSignalLabels()]
        signals: list[EDFSignal] = []
        for idx, label in enumerate(labels):
            if "annotation" in label.lower():
                continue
            fs = float(reader.getSampleFrequency(idx))
            data = np.asarray(reader.readSignal(idx), dtype=np.float32)
            signals.append(EDFSignal(label=label, fs=fs, data=data))

        start_time = _format_start_time(reader.getStartdatetime())
        annotations = _read_annotations(reader)
    return EDFRecord(signals=signals, start_time=start_time, annotations=annotations)


def _format_start_time(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.replace(microsecond=0).isoformat()
    return None


def _read_annotations(reader: "pyedflib.EdfReader") -> list[tuple[float, float, str]]:
    onsets, durations, labels = reader.readAnnotations()
    annotations: list[tuple[float, float, str]] = []
    for onset, duration, label in zip(onsets, durations, labels, strict=False):
        text = str(label).strip()
        if not text:
            continue
        annotations.append((float(onset), float(duration), text))
    return annotations


def _read_edf_record_legacy(path: Path) -> EDFRecord:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Invalid EDF header in {path}")

        start_date = fixed[168:176].decode("ascii", errors="ignore").strip()
        start_clock = fixed[176:184].decode("ascii", errors="ignore").strip()
        start_time = _normalize_start_time(start_date, start_clock)

        n_records = _parse_int(fixed[236:244])
        record_duration = _parse_float(fixed[244:252])
        n_signals = _parse_int(fixed[252:256])
        if n_signals <= 0:
            raise ValueError(f"No EDF signals in {path}")

        signal_header = handle.read(256 * n_signals)
        if len(signal_header) != 256 * n_signals:
            raise ValueError(f"Corrupt EDF signal header in {path}")

        fields = _parse_signal_header(signal_header, n_signals)
        labels = fields[0]
        phys_min = np.array([_parse_float(v.encode("ascii")) for v in fields[3]])
        phys_max = np.array([_parse_float(v.encode("ascii")) for v in fields[4]])
        dig_min = np.array([_parse_float(v.encode("ascii")) for v in fields[5]])
        dig_max = np.array([_parse_float(v.encode("ascii")) for v in fields[6]])
        samples_per_record = np.array(
            [_parse_int(v.encode("ascii")) for v in fields[8]], dtype=np.int64
        )

        if record_duration <= 0:
            raise ValueError(f"Invalid EDF record duration in {path}")

        header_bytes = 256 + 256 * n_signals
        file_size = path.stat().st_size
        bytes_per_record = int(np.sum(samples_per_record) * 2)
        if n_records < 0:
            if file_size <= header_bytes or bytes_per_record <= 0:
                raise ValueError(f"Unable to infer EDF record count in {path}")
            n_records = (file_size - header_bytes) // bytes_per_record

        total_samples = samples_per_record * n_records
        digital = [np.zeros(int(count), dtype=np.int16) for count in total_samples]
        offsets = np.zeros(n_signals, dtype=np.int64)

        for _ in range(n_records):
            for idx in range(n_signals):
                count = int(samples_per_record[idx])
                chunk = np.fromfile(handle, dtype="<i2", count=count)
                if chunk.size != count:
                    raise ValueError(f"Unexpected EOF while reading {path}")
                start = int(offsets[idx])
                end = start + count
                digital[idx][start:end] = chunk
                offsets[idx] = end

        signals: list[EDFSignal] = []
        annotations: list[tuple[float, float, str]] = []
        for idx in range(n_signals):
            label = labels[idx]
            if "annotation" in label.lower():
                annotations.extend(_parse_annotation_track(digital[idx]))
                continue
            scale_den = dig_max[idx] - dig_min[idx]
            if scale_den == 0:
                data = digital[idx].astype(np.float32)
            else:
                scale = (phys_max[idx] - phys_min[idx]) / scale_den
                data = (digital[idx] - dig_min[idx]) * scale + phys_min[idx]
                data = data.astype(np.float32)
            fs = float(samples_per_record[idx] / record_duration)
            signals.append(EDFSignal(label=label, fs=fs, data=data))
        return EDFRecord(signals=signals, start_time=start_time, annotations=annotations)


def _parse_signal_header(header: bytes, n_signals: int) -> list[list[str]]:
    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    cursor = 0
    fields: list[list[str]] = []
    for width in widths:
        values: list[str] = []
        for _ in range(n_signals):
            end = cursor + width
            values.append(header[cursor:end].decode("ascii", errors="ignore").strip())
            cursor = end
        fields.append(values)
    return fields


def _parse_int(raw: bytes) -> int:
    text = raw.decode("ascii", errors="ignore").strip()
    if not text:
        return 0
    return int(float(text))


def _parse_float(raw: bytes) -> float:
    text = raw.decode("ascii", errors="ignore").strip()
    if not text:
        return 0.0
    return float(text)


def _normalize_start_time(start_date: str, start_clock: str) -> str | None:
    if not start_clock:
        return None
    parts = start_clock.replace(".", ":").split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
    except ValueError:
        return None
    if not start_date:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    date_parts = start_date.split(".")
    if len(date_parts) != 3:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    try:
        day = int(date_parts[0])
        month = int(date_parts[1])
        year = int(date_parts[2])
    except ValueError:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    year += 1900 if year >= 85 else 2000
    return f"{year:04d}-{month:02d}-{day:02d}T{hours:02d}:{minutes:02d}:{seconds:02d}"


def _parse_annotation_track(digital: np.ndarray) -> list[tuple[float, float, str]]:
    raw = digital.astype("<i2").tobytes().replace(b"\x00\x00", b"\x00")
    text = raw.decode("latin-1", errors="ignore")
    entries = [chunk for chunk in text.split("\x00") if chunk]
    annotations: list[tuple[float, float, str]] = []
    for entry in entries:
        fields = [token for token in entry.split("\x14") if token]
        if not fields:
            continue
        onset = _safe_float(fields[0])
        duration = 0.0
        label = ""
        if len(fields) >= 2:
            if fields[1].startswith("\x15"):
                duration = _safe_float(fields[1][1:])
                label = fields[2] if len(fields) >= 3 else ""
            else:
                label = fields[1]
        label = label.replace("\x15", "").strip()
        if label:
            annotations.append((onset, duration, label))
    return annotations


def _safe_float(value: str) -> float:
    try:
        return float(value.strip())
    except Exception:
        return 0.0
