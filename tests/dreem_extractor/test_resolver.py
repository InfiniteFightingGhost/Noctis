import h5py

from dreem_extractor.config import load_config
from dreem_extractor.channels.resolver import resolve_channels


def test_resolve_ecg_path(tmp_path):
    path = tmp_path / "recording.h5"
    with h5py.File(path, "w") as h5file:
        signals = h5file.create_group("signals")
        emg = signals.create_group("emg")
        emg.create_dataset("ECG", data=[0, 1, 0])
        config = load_config()
        channel_map, _ = resolve_channels(h5file, config)
    assert channel_map["ecg"] == "/signals/emg/ECG"
