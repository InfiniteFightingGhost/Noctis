from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def profile_block(name: str, **extra: object) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logging.getLogger("app").info(
            "profile_block",
            extra={"block": name, "duration_seconds": duration, **extra},
        )
