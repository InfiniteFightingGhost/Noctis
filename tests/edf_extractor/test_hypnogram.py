from __future__ import annotations

import numpy as np

from edf_extractor.cli.main import _resolve_hypnogram_path
from edf_extractor.io.hypnogram import read_hypnogram


def test_read_hypnogram_parses_cap_events(tmp_path):
    cap_path = tmp_path / "recording.txt"
    cap_path.write_text(
        "0\t0\t22:00:00\tSleep-S0\t30\n"
        "0\t0\t22:00:30\tSleep-S1\t30\n"
        "0\t0\t22:01:00\tSleep-REM\t30\n",
        encoding="utf-8",
    )

    hypnogram, start_time, warnings = read_hypnogram(cap_path, epoch_sec=30)
    assert np.array_equal(hypnogram, np.array([0, 1, 4], dtype=np.int8))
    assert start_time == "22:00:00"
    assert warnings == []


def test_read_hypnogram_parses_isruc_text(tmp_path):
    isruc_path = tmp_path / "1_1.txt"
    isruc_path.write_text("0\n1\n2\n3\n5\n", encoding="utf-8")

    hypnogram, start_time, warnings = read_hypnogram(isruc_path, epoch_sec=30)
    assert np.array_equal(hypnogram, np.array([0, 1, 2, 3, 4], dtype=np.int8))
    assert start_time is None
    assert warnings == []


def test_read_hypnogram_parses_remlogic_columns(tmp_path):
    cap_path = tmp_path / "recording.txt"
    cap_path.write_text(
        "Sleep Stage\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation\n"
        "W\t22:30:28\tSLEEP-S0\t30\tEMG1-EMG2\n"
        "1\t22:30:58\tSLEEP-S1\t30\tEMG1-EMG2\n",
        encoding="utf-8",
    )

    hypnogram, start_time, warnings = read_hypnogram(cap_path, epoch_sec=30)
    assert np.array_equal(hypnogram, np.array([0, 1], dtype=np.int8))
    assert start_time == "22:30:28"
    assert warnings == []


def test_resolve_hypnogram_path_prefers_isruc_scorer(tmp_path):
    rec_path = tmp_path / "1.rec"
    rec_path.write_bytes(b"dummy")
    scorer_1 = tmp_path / "1_1.txt"
    scorer_2 = tmp_path / "1_2.txt"
    scorer_1.write_text("0\n", encoding="utf-8")
    scorer_2.write_text("0\n", encoding="utf-8")

    resolved = _resolve_hypnogram_path(
        rec_path,
        cap_override=None,
        cap_dir=None,
        preferred_isruc_scorer=2,
    )
    assert resolved == scorer_2
