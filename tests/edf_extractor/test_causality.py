from __future__ import annotations

import numpy as np

from edf_extractor.features.utils import bandpass_filter


def test_bandpass_filter_is_prefix_causal() -> None:
    fs = 50.0
    samples = 600
    t = np.arange(samples, dtype=np.float32) / fs
    signal = np.sin(2 * np.pi * 1.0 * t) + 0.2 * np.sin(2 * np.pi * 7.0 * t)
    full = bandpass_filter(signal, fs=fs, low=0.5, high=5.0)

    for prefix_len in (120, 240, 360):
        prefix = bandpass_filter(signal[:prefix_len], fs=fs, low=0.5, high=5.0)
        assert np.allclose(prefix, full[:prefix_len], atol=1e-9)
