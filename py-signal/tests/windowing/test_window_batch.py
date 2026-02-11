import numpy as np
from src.windowing.window import window_batch
from src.windowing.config import WindowConfig


def test_simple_windowing():
    signals = {"x": np.arange(10, dtype=float), "y": np.arange(10, dtype=float)}

    cfg = WindowConfig(size_samples=4, stride_samples=2)
    windows = window_batch(signals, start_ts=0, cfg=cfg)
    assert len(windows) == 4
    assert windows[0].data["x"].tolist() == [0, 1, 2, 3]
    assert windows[1].data["y"].tolist() == [2, 3, 4, 5]


def test_too_short_signal():
    signal = {"x": np.arange(3, dtype=float)}
    cfg = WindowConfig(size_samples=4, stride_samples=2)
    windows = window_batch(signal, 0, cfg)
    assert windows == []
