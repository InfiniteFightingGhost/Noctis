from __future__ import annotations

from prometheus_client import Counter, Histogram


REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency",
    ["path"],
)

INFERENCE_DURATION = Histogram(
    "inference_duration_seconds",
    "Model inference duration",
)

MODEL_RELOAD_COUNT = Counter(
    "model_reload_total",
    "Model reload count",
)
