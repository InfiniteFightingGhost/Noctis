from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    reason: Literal["too_short", "too_noisy", "flat_signal", "missing_channel"] | None
