from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from extractor_hardened.errors import ExtractionError


@dataclass
class AlignmentDecision:
    mode: str
    samples_per_epoch: int
    expected_samples: int
    available_samples: int
    status: str
    reason: str | None


@dataclass
class AlignedSignal:
    segments: list[np.ndarray | None]
    epoch_valid: np.ndarray
    decision: AlignmentDecision


def align_signal_deterministic(
    data: np.ndarray,
    fs: float,
    n_epochs: int,
    epoch_sec: int,
    mode: str,
    epoch_offset: int = 0,
) -> AlignedSignal:
    allowed_modes = ["strict", "reconcile"]
    if mode not in allowed_modes:
        raise ExtractionError(
            code="E_CONTRACT_VIOLATION",
            message="Unsupported alignment mode",
            details={"mode": mode, "allowed_modes": allowed_modes},
        )

    samples_per_epoch = int(round(fs * epoch_sec))
    expected_samples = samples_per_epoch * n_epochs
    available_samples = int(data.size)

    reason: str | None = None
    status = "exact"
    if available_samples != expected_samples:
        if mode == "strict":
            raise ExtractionError(
                code="E_ALIGN_MISMATCH",
                message="Signal length does not match expected epoch length",
                details={
                    "expected_samples": expected_samples,
                    "available_samples": available_samples,
                    "fs": fs,
                    "n_epochs": n_epochs,
                    "epoch_sec": epoch_sec,
                },
            )
        status = "reconciled"
        reason = "short" if available_samples < expected_samples else "long"

    segments: list[np.ndarray | None] = []
    valid = np.zeros(n_epochs, dtype=bool)
    for idx in range(n_epochs):
        start = (idx + epoch_offset) * samples_per_epoch
        end = start + samples_per_epoch
        if start < 0 or end > available_samples:
            segments.append(None)
            continue
        segments.append(data[start:end])
        valid[idx] = True

    decision = AlignmentDecision(
        mode=mode,
        samples_per_epoch=samples_per_epoch,
        expected_samples=expected_samples,
        available_samples=available_samples,
        status=status,
        reason=reason,
    )
    return AlignedSignal(segments=segments, epoch_valid=valid, decision=decision)
