from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from app.services.windowing import WindowedEpoch, build_windows


def test_build_windows_contiguous() -> None:
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    epochs = []
    for i in range(21):
        epochs.append(
            WindowedEpoch(
                epoch_index=i,
                epoch_start_ts=start + timedelta(seconds=30 * i),
                features=np.ones(10, dtype=np.float32) * i,
            )
        )
    windows = build_windows(epochs, window_size=21, allow_padding=False)
    assert len(windows) == 1
    assert windows[0].tensor.shape == (21, 10)
