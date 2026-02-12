# Noctis ML Inference Backend

Production-grade FastAPI + TimescaleDB inference service for sleep-stage prediction from 30s epoch features.

## Quickstart (Local)
1) Copy environment:
```
cp .env.example .env
```
2) Start the stack:
```
docker compose up --build
```
3) Verify health:
```
curl http://localhost:8000/healthz
```

## Quickstart (Dev)
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

## Multi-Tenant Model
- Every request is scoped by `tenant_id` in the JWT claims.
- Tenants are stored in `tenants` with `active`/`suspended` status.
- All data tables are tenant-scoped; cross-tenant reads return 404.

## Service-to-Service Auth
- Authentication uses JWTs with `Authorization: Bearer <token>`.
- Tokens must include: `sub` (service client id), `tenant_id`, `iss`, `aud`, `exp`.
- Service clients live in `service_clients` with roles: `ingest`, `read`, `admin`.
- Key rotation uses `service_client_keys` and `kid` headers in JWTs.

Provisioning (example SQL):
```
INSERT INTO service_clients (id, tenant_id, name, role, status, created_at)
VALUES ('<client-id>', '<tenant-id>', 'ingest-svc', 'ingest', 'active', NOW());

INSERT INTO service_client_keys (id, client_id, key_id, secret, status, created_at)
VALUES ('<key-id>', '<client-id>', '<kid>', '<shared-secret>', 'active', NOW());
```

Example token header + claim shape:
```
{
  "alg": "HS256",
  "kid": "<key-id>"
}
{
  "sub": "<service-client-id>",
  "tenant_id": "<tenant-id>",
  "iss": "noctis",
  "aud": "noctis-services",
  "exp": 1710000000
}
```

## API
- All endpoints (except health/metrics) require `Authorization: Bearer <token>`.
- POST `/v1/devices`
- POST `/v1/recordings`
- POST `/v1/epochs:ingest`
- POST `/v1/predict`
- GET `/v1/recordings/{id}/evaluation`
- GET `/v1/model/evaluation/global`
- GET `/v1/model/drift`
- GET `/v1/recordings/{id}`
- GET `/v1/recordings/{id}/epochs?from&to`
- GET `/v1/recordings/{id}/predictions?from&to`
- GET `/v1/recordings/{id}/summary?from&to`
- POST `/v1/models/reload`
- POST `/internal/stress/run`
- GET `/internal/performance`
- GET `/internal/performance/stats`
- GET `/internal/monitoring/summary`
- GET `/internal/timescale/policies`
- POST `/internal/timescale/dry-run`
- POST `/internal/timescale/apply`
- GET `/internal/audit/report`
- GET `/internal/faults`
- POST `/internal/faults/enable`
- POST `/internal/faults/disable`
- GET `/healthz`
- GET `/readyz`
- GET `/metrics`

## Documentation
- `docs/system-design.md`
- `docs/plan.md`
- `docs/ownership.md`
- `docs/autonomous-mode.md`
- `docs/definition-of-done.md`
- `docs/testing.md`
- `docs/api-contract-map.md`

## Autonomous Runner (OpenCode)
Preflight before starting orchestration:
```
command -v opencode && opencode --version
```

Policy rules:
- `command -v opencode && opencode --version` MUST pass before any autonomous orchestration run.
- Builder phases MUST execute sequentially in configured order (no parallel builder execution).

Create project agents if needed:
```
opencode agent create
```

Run autonomous orchestration:
```
python tools/run_autonomous.py --config docs/runner-config.json
```

Notes:
- Builder phases are executed sequentially by the default runner contract.
- Runner outputs are attempt-qualified under `docs/run-log/outputs/` and do not overwrite retries.

## Architecture
```
          +--------------------+
          |  FastAPI Routers   |
          | /v1  /internal     |
          +---------+----------+
                    |
        +-----------+-------------+
        |     Service Layer       |
        | eval  drift  stress     |
        | perf  monitoring        |
        +-----------+-------------+
                    |
        +-----------+-------------+
        |  SQLAlchemy + Timescale |
        |  epochs/predictions     |
        |  eval/usage/feature     |
        +-----------+-------------+
                    |
        +-----------+-------------+
        |   Model Registry        |
        |  models/<version>       |
        +-------------------------+
```

## Monitoring
- Prometheus metrics: `/metrics` (includes inference, errors, DB retries, drift/eval counters)
- Key metrics: `window_build_duration_seconds`, `prediction_confidence_histogram`, `drift_score_gauge`,
  `drift_severity_gauge`, `device_ingest_rate`, `ingest_failures_total`,
  `db_commit_latency_seconds`, `inference_duration_seconds`,
  `inference_p95_latency_ms`, `inference_p99_latency_ms`, `model_reload_success_total`,
  `active_tenant_count`
- Correlation IDs: request and response use `X-Request-Id` / `X-Correlation-Id`
- Internal snapshots: `/internal/performance`, `/internal/monitoring/summary` (admin role)
- `/internal/performance` includes inference timing and DB write speed summaries
- `/internal/performance/stats` includes optional `slow_queries` when `pg_stat_statements` is available

## SLOs
- p95 inference latency: `SLO_INFERENCE_P95_MS` (default 350ms)
- p99 inference latency: `SLO_INFERENCE_P99_MS` (default 700ms)
- ingest failure rate: `SLO_INGEST_FAILURE_RATE` (default 1%)
- DB commit p95 latency: `SLO_DB_COMMIT_P95_MS` (default 200ms)
- model reload success rate: `SLO_MODEL_RELOAD_SUCCESS_RATE` (default 99%)

## Drift Guide
- `/v1/model/drift` compares recent vs baseline windows
- Metrics: PSI + KL on stage distribution, Z-score on confidence mean, feature-level Z-scores
- Thresholds:
  - `DRIFT_PSI_THRESHOLD` (default 0.2)
  - `DRIFT_KL_THRESHOLD` (default 0.1)
  - `DRIFT_Z_THRESHOLD` (default 3.0)
- Severity: LOW (< threshold), MEDIUM (>= threshold), HIGH (>= 2x threshold)
- `flagged_features` lists feature indices exceeding Z-score thresholds

## Load Testing
Run reproducible stress loops against the active model:
```
curl -X POST http://localhost:8000/internal/stress/run \
  -H "Authorization: Bearer <admin-jwt>" \
  -H "Content-Type: application/json" \
  -d '{"mode":"both","device_count":50,"hours":8,"ingest_batch_size":256,"iterations":200,"batch_size":32,"seed":42}'
```

Load test script (batch ingestion + concurrent predict + reload under traffic):
```
python scripts/load_test.py --base-url http://localhost:8000 --api-token <jwt> --admin-token <admin-jwt> \
  --device-count 100 --hours 8 --predict-concurrency 20 --seed 42 --reload-during
```

## Hardening
- Request timeouts via `REQUEST_TIMEOUT_SECONDS`
- DB retry policy and circuit breaker (`DB_RETRY_*`, `DB_CIRCUIT_*`)
- Safe model reload lock and structured error codes
- Graceful model unavailability response (HTTP 503)
- Size DB pools (`DB_POOL_SIZE`, `DB_MAX_OVERFLOW`) for ingest burst capacity
- Use Timescale retention/compression policies for long-running deployments
- Monitor drift severity and alert on `flagged_features`
- Use rolling 7-day evaluation for trend checks

## Failure Scenarios
- DB unavailable: circuit breaker opens, 503 with `db_circuit_open` / `db_error`
- Model unavailable: 503 with `model_unavailable`
- Timeout: 504 with `request_timeout`
- Validation failures: 4xx with `validation_error` / `bad_request`
- Model reload during inference: requests wait for reload lock, then proceed safely
- Drift spikes: metrics include severity and flagged features for investigation
- Fault injection: use `/internal/faults/*` (admin role) to simulate DB outages, latency, model unavailability, or timeouts

## Model Registry
Models live under `models/<version>/`:
- `weights.npy`, `bias.npy`
- `label_map.json`
- `feature_schema.json`
- `metadata.json`

Update `ACTIVE_MODEL_VERSION` in `.env` to switch versions. Call `/v1/models/reload` to reload without restart.

## ML Lifecycle (Guidance)
Diagram description (happy path):
1) Ingest epochs via `/v1/epochs:ingest` into Timescale.
2) Build a training dataset from stored epochs + ground-truth labels.
3) Train offline to produce model artifacts in `models/<version>/`.
4) Validate against holdout + drift/eval baselines.
5) Promote by updating `ACTIVE_MODEL_VERSION` and calling `/v1/models/reload`.
6) Replay historical windows to compare predictions vs labels and monitor drift.
7) Retrain when drift/eval thresholds or performance regressions are sustained.

Training instructions (offline):
- Use the canonical feature order from `feature_schema.json` and record the version in `metadata.json`.
- Produce the full artifact set (`weights.npy`, `bias.npy`, `label_map.json`, `feature_schema.json`, `metadata.json`).
- Keep training deterministic (fixed seeds, pinned data snapshot), and log metrics for promotion review.

Dataset builder expectations:
- Source epochs from Timescale (or `/v1/recordings/{id}/epochs?from&to`) and join with labels from your ground-truth system.
- Enforce feature schema version and ordering at build time; reject mismatches.
- Split by recording/device to avoid leakage and keep a stable holdout.

Promotion checklist:
- Run evaluation endpoints (`/v1/recordings/{id}/evaluation`, `/v1/model/evaluation/global`) against the candidate.
- Record drift baselines and thresholds; update monitoring dashboards if needed.
- Promote by switching `ACTIVE_MODEL_VERSION`, reload, and verify `/readyz` + `/v1/model/drift`.

Replay workflow:
- For a target window, fetch epochs (`/v1/recordings/{id}/epochs?from&to`) and call `/v1/predict` in batches.
- Compare predictions vs labels and log deltas to assess regression risk.

Retraining workflow:
- Trigger on sustained drift (MEDIUM/HIGH), evaluation regressions, or data distribution shifts.
- Rebuild dataset, retrain, validate, and run replay before promotion.

Failure recovery:
- If reload fails or metrics regress, roll back by restoring the prior `ACTIVE_MODEL_VERSION` and reloading.
- If DB is degraded, defer replay/retraining and use `/internal/monitoring/summary` to confirm recovery.

## Feature Schema Versioning
`feature_schema.json` defines a version string and feature order. Ingestion and prediction reject mismatches.
The server decodes list, dict, or packed base64 float32 payloads into the canonical float32 vector.

## TimescaleDB Policies
Retention, compression, and continuous aggregates are managed via:
- `POST /internal/timescale/dry-run` (preview actions)
- `POST /internal/timescale/apply` (apply policies)
- `GET /internal/timescale/policies` (current state)

Defaults (override via env):
- `EPOCHS_RETENTION_DAYS=90`
- `PREDICTIONS_RETENTION_DAYS=180`
- `EPOCHS_COMPRESSION_AFTER_DAYS=7`

Continuous aggregates:
- `recording_daily_summary` (per recording, daily)
- `device_daily_summary` (per device, daily)

## Backup & Restore
- Create logical backup: `python -m app.ops.backup --output backups/noctis.dump`
- Validate restore: `python -m app.ops.restore_test backups/noctis.dump`
- Restore validation spins a temporary DB, restores, and compares row counts + checksums.

## Operational Runbooks
- Auth failure spikes: inspect `auth_failures_total`, verify JWT issuer/audience, rotate `service_client_keys`.
- Ingest failures: check `ingest_failures_total`, DB commit latency, and hypertable chunk pressure.
- Drift alerts: inspect `drift_severity_gauge`, run replay to confirm, schedule retrain.
- Backup validation failures: re-run `restore_test` and inspect checksum deltas per table.

## Tests
Unit tests:
```
pytest tests/unit
```
Integration tests (requires TimescaleDB):
```
export INTEGRATION_TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/noctis
pytest tests/integration
```

## Production
```
docker compose up --build -d
```
Gunicorn uses `docker/gunicorn.conf.py` with Uvicorn workers.

## Troubleshooting
- If `/readyz` fails, confirm TimescaleDB is reachable and `alembic upgrade head` ran.
- If `/v1/predict` returns feature schema mismatch, verify `FEATURE_SCHEMA_VERSION` and model schema.
- If ingestion dedupes unexpectedly, check `(recording_id, epoch_index)` uniqueness.
