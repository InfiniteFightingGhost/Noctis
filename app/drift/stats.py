from __future__ import annotations

from math import log, sqrt


def _normalize(dist: dict[str, float], eps: float = 1e-8) -> dict[str, float]:
    total = sum(dist.values())
    if total <= 0:
        return {k: eps for k in dist}
    return {k: max(eps, v / total) for k, v in dist.items()}


def psi(current: dict[str, float], baseline: dict[str, float]) -> float:
    cur = _normalize(current)
    base = _normalize(baseline)
    keys = set(cur) | set(base)
    value = 0.0
    for key in keys:
        p = cur.get(key, 1e-8)
        q = base.get(key, 1e-8)
        value += (p - q) * log(p / q)
    return value


def kl_divergence(current: dict[str, float], baseline: dict[str, float]) -> float:
    cur = _normalize(current)
    base = _normalize(baseline)
    keys = set(cur) | set(base)
    value = 0.0
    for key in keys:
        p = cur.get(key, 1e-8)
        q = base.get(key, 1e-8)
        value += p * log(p / q)
    return value


def z_score(current_mean: float, baseline_mean: float, baseline_std: float) -> float:
    if baseline_std <= 0:
        return 0.0
    return (current_mean - baseline_mean) / baseline_std


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
    return sqrt(variance)
