from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitRule:
    path: str
    max_requests: int


class SlidingWindowRateLimiter:
    def __init__(
        self, *, window_seconds: int, default_limit: int, rules: list[RateLimitRule]
    ) -> None:
        self._window_seconds = max(int(window_seconds), 1)
        self._default_limit = max(int(default_limit), 1)
        self._rules = sorted(rules, key=lambda rule: len(rule.path), reverse=True)
        self._lock = threading.Lock()
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, *, client_id: str, path: str) -> bool:
        limit = self._limit_for_path(path)
        now = time.monotonic()
        threshold = now - self._window_seconds
        key = f"{client_id}:{path}"
        with self._lock:
            bucket = self._requests[key]
            while bucket and bucket[0] <= threshold:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
        return True

    def _limit_for_path(self, path: str) -> int:
        for rule in self._rules:
            if path.startswith(rule.path):
                return max(int(rule.max_requests), 1)
        return self._default_limit
