from __future__ import annotations

import numpy as np

from extractor_hardened.alignment import align_signal_deterministic
from extractor_hardened.contracts import load_contracts


def align_signal(
    data: np.ndarray,
    fs: float,
    n_epochs: int,
    epoch_sec: int,
    epoch_offset: int = 0,
) -> tuple[list[np.ndarray | None], list[str]]:
    mode = str(load_contracts().alignment_policy.get("mode", "reconcile"))
    aligned = align_signal_deterministic(
        data,
        fs,
        n_epochs,
        epoch_sec,
        mode=mode,
        epoch_offset=epoch_offset,
    )
    warnings: list[str] = []
    if aligned.decision.status != "exact":
        warnings.append("signal_length_reconciled")
    return aligned.segments, warnings
