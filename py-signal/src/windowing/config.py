from dataclasses import dataclass


@dataclass(frozen=True)
class WindowConfig:
    size_samples: int
    stride_samples: int
