import numpy as np

from edf_extractor.config import load_config
from edf_extractor.pipeline import extract_record

from tests.edf_extractor.test_e2e import _write_cap, _write_test_edf


def test_alignment_warning_on_short_signal(tmp_path):
    edf_path = tmp_path / "recording.edf"
    cap_path = tmp_path / "recording.txt"

    fs = 20
    epoch_sec = 30
    n_epochs_signal = 2
    n_epochs_hypnogram = 3
    samples = fs * epoch_sec * n_epochs_signal

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
        ],
    )

    config = load_config()
    result = extract_record(edf_path, cap_path, config)
    assert "signal_length_reconciled" in result.warnings
    assert len(result.hypnogram) == n_epochs_hypnogram


def test_applies_cap_edf_start_offset(tmp_path):
    edf_path = tmp_path / "recording.edf"
    cap_path = tmp_path / "recording.txt"

    fs = 20
    n_epochs = 2
    samples = fs * 30 * n_epochs
    t = np.arange(samples, dtype=np.float32) / fs
    ecg = 0.1 * np.sin(2.0 * np.pi * 1.0 * t)
    ecg[np.arange(0, samples, fs)] += 4.0

    _write_test_edf(edf_path, [("ECG", ecg, fs)], start_clock="00.01.00")
    _write_cap(
        cap_path,
        [
            ("00:00:00", "Sleep-S0", 30),
            ("00:00:30", "Sleep-S1", 30),
        ],
    )

    result = extract_record(edf_path, cap_path, load_config())
    assert "cap_edf_start_offset_applied" in result.warnings
    assert result.metadata["epoch_offset"] == -2
    assert not np.any(result.valid_mask)
