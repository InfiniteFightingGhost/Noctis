import struct

import numpy as np
import pytest

from edf_extractor.config import load_config
from edf_extractor.constants import FEATURE_ORDER, FlagBits
from edf_extractor.pipeline import extract_record
from extractor_hardened.errors import ExtractionError


def test_end_to_end_ecg_only(tmp_path):
    edf_path = tmp_path / "recording.edf"
    cap_path = tmp_path / "recording.txt"

    fs = 100
    epoch_sec = 30
    n_epochs = 4
    samples = fs * epoch_sec * n_epochs
    t = np.arange(samples, dtype=np.float32) / fs
    ecg = 0.1 * np.sin(2.0 * np.pi * 1.0 * t)
    ecg[np.arange(0, samples, fs)] += 4.0

    _write_test_edf(edf_path, [("ECG", ecg, fs)])
    _write_cap(
        cap_path,
        [
            ("00:00:00", "Sleep-S0", 30),
            ("00:00:30", "Sleep-S1", 30),
            ("00:01:00", "Sleep-S2", 30),
            ("00:01:30", "Sleep-Unscored", 30),
        ],
    )

    config = load_config()
    result = extract_record(edf_path, cap_path, config)

    assert result.features.shape == (n_epochs, len(FEATURE_ORDER))
    flags = result.features[:, FEATURE_ORDER.index("flags")]
    assert np.all((flags & (1 << FlagBits.ECG_PRESENT)) > 0)
    assert np.all((flags & (1 << FlagBits.RR_FROM_EDR)) > 0)
    assert (flags[3] & (1 << FlagBits.STAGE_SCORED)) == 0
    assert not result.valid_mask[3]


def test_csv_cap_and_duplicate_label_index_stability(tmp_path):
    edf_path = tmp_path / "recording.edf"
    cap_path = tmp_path / "recording.csv"

    fs = 100
    n_epochs = 3
    samples = fs * 30 * n_epochs
    t = np.arange(samples, dtype=np.float32) / fs
    ecg_good = 0.1 * np.sin(2.0 * np.pi * 1.0 * t)
    ecg_good[np.arange(0, samples, fs)] += 4.0
    ecg_bad = np.zeros(samples, dtype=np.float32)

    _write_test_edf(edf_path, [("ECG", ecg_good, fs), ("ECG", ecg_bad, fs)])
    _write_cap(
        cap_path,
        [
            ("00:00:00", "Sleep-S0", 30),
            ("00:00:30", "Sleep-S1", 30),
            ("00:01:00", "Sleep-S2", 30),
        ],
        delimiter=",",
    )

    result = extract_record(edf_path, cap_path, load_config())
    flags = result.features[:, FEATURE_ORDER.index("flags")]
    assert np.all((flags & (1 << FlagBits.ECG_PRESENT)) > 0)


def _write_cap(path, rows, delimiter="\t"):
    with path.open("w", encoding="utf-8") as handle:
        for clock, event, duration in rows:
            handle.write(delimiter.join(["0", "0", clock, event, str(duration)]) + "\n")


def _write_test_edf(path, channels, start_clock="00.00.00"):
    n_signals = len(channels)
    samples_per_record = [int(fs) for _, _, fs in channels]
    n_records = max(
        int(np.ceil(len(data) / spr))
        for (_, data, _), spr in zip(channels, samples_per_record, strict=True)
    )
    record_duration = 1

    labels = [_pad(label, 16) for label, _, _ in channels]
    transducer = [_pad("", 80) for _ in channels]
    phys_dim = [_pad("uV", 8) for _ in channels]
    phys_min = [_pad("-32768", 8) for _ in channels]
    phys_max = [_pad("32767", 8) for _ in channels]
    dig_min = [_pad("-32768", 8) for _ in channels]
    dig_max = [_pad("32767", 8) for _ in channels]
    prefilter = [_pad("", 80) for _ in channels]
    spr = [_pad(str(value), 8) for value in samples_per_record]
    reserved_sig = [_pad("", 32) for _ in channels]

    fixed = b"".join(
        [
            _pad("0", 8).encode("ascii"),
            _pad("test", 80).encode("ascii"),
            _pad("test", 80).encode("ascii"),
            _pad("01.01.24", 8).encode("ascii"),
            _pad(start_clock, 8).encode("ascii"),
            _pad(str(256 + 256 * n_signals), 8).encode("ascii"),
            _pad("", 44).encode("ascii"),
            _pad(str(n_records), 8).encode("ascii"),
            _pad(str(record_duration), 8).encode("ascii"),
            _pad(str(n_signals), 4).encode("ascii"),
        ]
    )

    signal_header = b"".join(
        [
            "".join(labels).encode("ascii"),
            "".join(transducer).encode("ascii"),
            "".join(phys_dim).encode("ascii"),
            "".join(phys_min).encode("ascii"),
            "".join(phys_max).encode("ascii"),
            "".join(dig_min).encode("ascii"),
            "".join(dig_max).encode("ascii"),
            "".join(prefilter).encode("ascii"),
            "".join(spr).encode("ascii"),
            "".join(reserved_sig).encode("ascii"),
        ]
    )

    with path.open("wb") as handle:
        handle.write(fixed)
        handle.write(signal_header)
        for rec in range(n_records):
            for (_, data, _), record_samples in zip(channels, samples_per_record, strict=True):
                start = rec * record_samples
                end = start + record_samples
                segment = data[start:end]
                if len(segment) < record_samples:
                    padded = np.zeros(record_samples, dtype=np.float32)
                    padded[: len(segment)] = segment
                    segment = padded
                for value in segment:
                    handle.write(
                        struct.pack(
                            "<h",
                            int(np.clip(np.round(float(value) * 1000), -32768, 32767)),
                        )
                    )


def _pad(value, width):
    return str(value)[:width].ljust(width)
