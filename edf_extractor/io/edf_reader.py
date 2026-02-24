from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EDFSignal:
    label: str
    fs: float
    data: np.ndarray


def read_edf(path: str | Path) -> list[EDFSignal]:
    path = Path(path)
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Invalid EDF header in {path}")

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
        for idx in range(n_signals):
            scale_den = dig_max[idx] - dig_min[idx]
            if scale_den == 0:
                data = digital[idx].astype(np.float32)
            else:
                scale = (phys_max[idx] - phys_min[idx]) / scale_den
                data = (digital[idx] - dig_min[idx]) * scale + phys_min[idx]
                data = data.astype(np.float32)
            fs = float(samples_per_record[idx] / record_duration)
            signals.append(EDFSignal(label=labels[idx], fs=fs, data=data))
        return signals


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
