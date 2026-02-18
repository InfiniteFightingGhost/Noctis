import h5py
import numpy as np

from dreem_extractor.config import load_config
from dreem_extractor.pipeline import extract_record


def test_alignment_warning_on_short_signal(tmp_path):
    path = tmp_path / "recording.h5"
    fs = 10.0
    epoch_sec = 30
    samples = int(fs * epoch_sec * 2)
    ecg = np.zeros(samples, dtype=np.float32)
    ecg[::10] = 1.0
    hypnogram = np.array([0, 1, 2], dtype=np.int8)

    with h5py.File(path, "w") as h5file:
        h5file.create_dataset("hypnogram", data=hypnogram)
        signals = h5file.create_group("signals")
        emg = signals.create_group("emg")
        ds = emg.create_dataset("ECG", data=ecg)
        ds.attrs["sampling_rate"] = fs

    config = load_config()
    result = extract_record(path, config)
    assert "signal_shorter_than_hypnogram" in result.warnings
