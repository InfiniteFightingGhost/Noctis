from __future__ import annotations

import h5py
from dreem_extractor.channels.resolver import resolve_channels
from dreem_extractor.config import load_config


def test_resolver_prefers_deterministic_match_for_optional_ambiguity(tmp_path) -> None:
    path = tmp_path / "ambiguous.h5"
    with h5py.File(path, "w") as h5file:
        signals = h5file.create_group("signals")
        emg = signals.create_group("emg")
        emg.create_dataset("ECG", data=[1, 2, 3])
        emg.create_dataset("ecg", data=[1, 2, 3])
        eeg = signals.create_group("eeg")
        eeg.create_dataset("A", data=[1, 2, 3])
        eeg.create_dataset("B", data=[1, 2, 3])
        config = load_config()
        channel_map, _ = resolve_channels(h5file, config)
    assert channel_map["ecg"] == "/signals/emg/ECG"
    assert channel_map["eeg"] == "/signals/eeg/A"
