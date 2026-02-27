from __future__ import annotations

import numpy as np

from dreem_extractor.config import load_config
from dreem_extractor.features.base import FeatureContext
from dreem_extractor.features.hr_ecg import ECGHRPlugin
from dreem_extractor.features.utils import bandpass_filter, lowpass_filter
from dreem_extractor.models import SignalSeries


def test_hr_feature_is_prefix_causal() -> None:
    config = load_config()
    n_epochs = 5
    fs = 50.0
    epoch_sec = config.epoch_sec
    samples = int(fs * epoch_sec * n_epochs)
    t = np.arange(samples, dtype=np.float32) / fs
    ecg = 0.05 * np.sin(2 * np.pi * 1.0 * t)
    ecg[np.arange(0, samples, int(fs))] += 3.0

    full_ctx = FeatureContext(
        config=config,
        n_epochs=n_epochs,
        signals={"ecg": SignalSeries(name="ecg", data=ecg, fs=fs, segments=[])},
        hypnogram=np.zeros(n_epochs, dtype=np.int8),
    )
    plugin = ECGHRPlugin()
    full = plugin.compute(full_ctx).features["hr_mean"]

    for idx in range(n_epochs):
        prefix_samples = int(fs * epoch_sec * (idx + 1))
        prefix_ctx = FeatureContext(
            config=config,
            n_epochs=idx + 1,
            signals={
                "ecg": SignalSeries(name="ecg", data=ecg[:prefix_samples], fs=fs, segments=[])
            },
            hypnogram=np.zeros(idx + 1, dtype=np.int8),
        )
        prefix = plugin.compute(prefix_ctx).features["hr_mean"]
        assert np.array_equal(prefix, full[: idx + 1])


def test_bandpass_filter_is_prefix_causal() -> None:
    fs = 50.0
    samples = 600
    t = np.arange(samples, dtype=np.float32) / fs
    signal = np.sin(2 * np.pi * 1.0 * t) + 0.2 * np.sin(2 * np.pi * 8.0 * t)
    full = bandpass_filter(signal, fs=fs, low=0.5, high=5.0)

    for prefix_len in (120, 240, 360):
        prefix = bandpass_filter(signal[:prefix_len], fs=fs, low=0.5, high=5.0)
        assert np.allclose(prefix, full[:prefix_len], atol=1e-9)


def test_lowpass_filter_is_prefix_causal() -> None:
    fs = 50.0
    samples = 600
    t = np.arange(samples, dtype=np.float32) / fs
    signal = np.sin(2 * np.pi * 1.5 * t) + 0.2 * np.sin(2 * np.pi * 9.0 * t)
    full = lowpass_filter(signal, fs=fs, cutoff=4.0)

    for prefix_len in (120, 240, 360):
        prefix = lowpass_filter(signal[:prefix_len], fs=fs, cutoff=4.0)
        assert np.allclose(prefix, full[:prefix_len], atol=1e-9)
