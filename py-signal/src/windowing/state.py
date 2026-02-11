from dataclasses import dataclass
import numpy as np


@dataclass
class WindowState:
    buffer: np.ndarray  # shape (N, C)
