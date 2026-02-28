from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge


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

INFERENCE_P95_MS = Gauge(
    "inference_p95_latency_ms",
    "P95 inference latency in milliseconds",
)

INFERENCE_P99_MS = Gauge(
    "inference_p99_latency_ms",
    "P99 inference latency in milliseconds",
)

WINDOW_BUILD_DURATION = Histogram(
    "window_build_duration_seconds",
    "Window build duration",
)

MODEL_RELOAD_COUNT = Counter(
    "model_reload_total",
    "Model reload count",
)

MODEL_RELOAD_SUCCESS = Counter(
    "model_reload_success_total",
    "Model reload success count",
)

MODEL_RELOAD_FAILURE = Counter(
    "model_reload_failure_total",
    "Model reload failure count",
)

DB_RETRY_COUNT = Counter(
    "db_retry_total",
    "Database retry attempts",
    ["operation"],
)

DB_CIRCUIT_OPEN = Counter(
    "db_circuit_open_total",
    "Database circuit breaker open events",
)

MODEL_UNAVAILABLE_COUNT = Counter(
    "model_unavailable_total",
    "Model unavailable responses",
)

ERROR_COUNT = Counter(
    "error_total",
    "Structured error responses",
    ["code", "classification"],
)

EVALUATION_REQUESTS = Counter(
    "evaluation_requests_total",
    "Evaluation endpoint requests",
    ["scope"],
)

DRIFT_REQUESTS = Counter(
    "drift_requests_total",
    "Drift endpoint requests",
)

DRIFT_SCORE_GAUGE = Gauge(
    "drift_score_gauge",
    "Latest drift score by metric",
    ["metric"],
)

DRIFT_SEVERITY_GAUGE = Gauge(
    "drift_severity_gauge",
    "Latest drift severity level",
    ["scope"],
)

STRESS_RUNS = Counter(
    "stress_runs_total",
    "Stress runs executed",
)

PREDICTION_CONFIDENCE_HISTOGRAM = Histogram(
    "prediction_confidence_histogram",
    "Prediction confidence distribution",
)

DEVICE_INGEST_RATE = Gauge(
    "device_ingest_rate",
    "Latest device ingest rate (epochs/sec)",
)

INGEST_REQUESTS = Counter(
    "ingest_requests_total",
    "Total ingest requests",
)

INGEST_FAILURES = Counter(
    "ingest_failures_total",
    "Failed ingest requests",
)

DB_COMMIT_LATENCY = Histogram(
    "db_commit_latency_seconds",
    "Database commit latency",
)

MEMORY_RSS_MB = Gauge(
    "process_memory_rss_mb",
    "Process RSS memory in MB",
)

AUTH_FAILURE_COUNT = Counter(
    "auth_failures_total",
    "Authorization failures",
    ["reason"],
)

RATE_LIMITED_COUNT = Counter(
    "rate_limited_requests_total",
    "Requests rejected by rate limiting",
    ["path"],
)

ACTIVE_TENANT_COUNT = Gauge(
    "active_tenant_count",
    "Number of active tenants",
)
