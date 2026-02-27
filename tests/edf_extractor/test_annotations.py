from __future__ import annotations

import numpy as np

from edf_extractor.io.hypnogram import merge_hypnogram_tracks


def test_annotation_merge_deterministic_conflict_handling() -> None:
    cap = np.array([0, 1, 2, 2], dtype=np.int8)
    annotations = [
        (30.0, 30.0, "Sleep-S2"),
        (60.0, 30.0, "Sleep-REM"),
        (60.0, 30.0, "Sleep-S2"),
    ]
    merged_a, warnings_a = merge_hypnogram_tracks(cap, 30, annotations)
    merged_b, warnings_b = merge_hypnogram_tracks(cap, 30, list(reversed(annotations)))
    assert np.array_equal(merged_a, merged_b)
    assert warnings_a == warnings_b
    assert "annotation_cap_conflict" in warnings_a
