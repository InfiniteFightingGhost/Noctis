from __future__ import annotations

import numpy as np
import pytest

from extractor_hardened.alignment import align_signal_deterministic
from extractor_hardened.errors import ExtractionError


def test_alignment_strict_fails_on_mismatch() -> None:
    data = np.zeros(50, dtype=np.float32)
    with pytest.raises(ExtractionError) as exc:
        align_signal_deterministic(data, fs=10.0, n_epochs=6, epoch_sec=1, mode="strict")
    assert exc.value.code == "E_ALIGN_MISMATCH"


def test_alignment_reconcile_marks_invalid_epochs() -> None:
    data = np.zeros(50, dtype=np.float32)
    aligned = align_signal_deterministic(data, fs=10.0, n_epochs=6, epoch_sec=1, mode="reconcile")
    assert aligned.decision.status == "reconciled"
    assert int(np.sum(aligned.epoch_valid)) == 5


def test_alignment_fuzz_reconcile_deterministic() -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=137).astype(np.float32)
    left = align_signal_deterministic(data, fs=10.0, n_epochs=15, epoch_sec=1, mode="reconcile")
    right = align_signal_deterministic(data, fs=10.0, n_epochs=15, epoch_sec=1, mode="reconcile")
    assert left.decision == right.decision
    assert np.array_equal(left.epoch_valid, right.epoch_valid)


def test_alignment_invalid_mode_rejected() -> None:
    data = np.zeros(60, dtype=np.float32)
    with pytest.raises(ExtractionError) as exc:
        align_signal_deterministic(data, fs=10.0, n_epochs=6, epoch_sec=1, mode="unknown")
    assert exc.value.code == "E_CONTRACT_VIOLATION"
    assert exc.value.details == {
        "mode": "unknown",
        "allowed_modes": ["strict", "reconcile"],
    }
