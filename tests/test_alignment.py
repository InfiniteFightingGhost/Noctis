import h5py
import numpy as np

from extractor.config import ExtractConfig, UINT8_UNKNOWN
from extractor.extract import extract_recording


def test_alignment_marks_missing_epochs(tmp_path):
    path = tmp_path / "recording.h5"
    hypnogram = np.array([0, 1, 2], dtype=np.int16)
    fs = 10.0
    epoch_sec = 1
    samples = int(fs * epoch_sec * 2)
    ecg = np.zeros(samples, dtype=np.float32)
    ecg[::10] = 1.0

    with h5py.File(path, "w") as h5file:
        h5file.create_dataset("hypnogram", data=hypnogram)
        signals = h5file.create_group("signals")
        emg = signals.create_group("emg")
        ds = emg.create_dataset("ECG", data=ecg)
        ds.attrs["sampling_rate"] = fs

    result = extract_recording(path, ExtractConfig(epoch_sec=epoch_sec))
    assert len(result.records) == len(hypnogram)
    assert result.features["hr_mean"][2] == UINT8_UNKNOWN
    assert result.features["flags"][2] & 1 == 0
