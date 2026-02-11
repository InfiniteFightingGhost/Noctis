from dataclasses import dataclass
import numpy as np
from typing import Dict

Signal = np.ndarray
MultiSignal = Dict[str, Signal]


@dataclass(frozen=True)
class Window:
    start_ts: int
    end_ts: int
    data: MultiSignal
