import h5py
import numpy as np

from dreem_extractor.config import load_config
from dreem_extractor.constants import FlagBits, FEATURE_ORDER
from dreem_extractor.pipeline import extract_record


def test_end_to_end_ecg_only(tmp_path):
    path = tmp_path / "recording.h5"
    fs = 100.0
    epoch_sec = 30
    n_epochs = 4
    samples = int(fs * epoch_sec * n_epochs)
    t = np.arange(samples) / fs
    ecg = 0.01 * np.sin(2 * np.pi * 1.0 * t)
    peaks = (np.arange(samples // int(fs)) * fs).astype(int)
    ecg[peaks] += 1.0
    hypnogram = np.array([0, 1, 2, -1], dtype=np.int8)

    with h5py.File(path, "w") as h5file:
        h5file.create_dataset("hypnogram", data=hypnogram)
        signals = h5file.create_group("signals")
        emg = signals.create_group("emg")
        ds = emg.create_dataset("ECG", data=ecg.astype(np.float32))
        ds.attrs["sampling_rate"] = fs

    config = load_config()
    result = extract_record(path, config)
    assert result.features.shape == (n_epochs, len(FEATURE_ORDER))
    flags = result.features[:, FEATURE_ORDER.index("flags")]
    assert np.all((flags & (1 << FlagBits.ECG_PRESENT)) > 0)
    assert np.all((flags & (1 << FlagBits.RR_FROM_EDR)) > 0)
    assert (flags[3] & (1 << FlagBits.STAGE_SCORED)) == 0
    assert not result.valid_mask[3]
