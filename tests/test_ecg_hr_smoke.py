import h5py
import numpy as np

from extractor.config import ExtractConfig, UINT8_UNKNOWN
from extractor.extract import extract_recording


def test_ecg_hr_smoke(tmp_path):
    path = tmp_path / "recording.h5"
    fs = 100.0
    epoch_sec = 30
    duration_sec = 60
    samples = int(fs * duration_sec)
    t = np.arange(samples) / fs
    ecg = 0.05 * np.sin(2 * np.pi * 1.0 * t)
    peak_indices = (np.arange(duration_sec) * fs).astype(int)
    ecg[peak_indices] += 1.0
    hypnogram = np.array([0, 0], dtype=np.int16)

    with h5py.File(path, "w") as h5file:
        h5file.create_dataset("hypnogram", data=hypnogram)
        signals = h5file.create_group("signals")
        emg = signals.create_group("emg")
        ds = emg.create_dataset("ECG", data=ecg.astype(np.float32))
        ds.attrs["sampling_rate"] = fs

    result = extract_recording(path, ExtractConfig(epoch_sec=epoch_sec))
    hr_mean = result.features["hr_mean"][0]
    assert hr_mean != UINT8_UNKNOWN
    assert 55 <= hr_mean <= 65
