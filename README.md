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

## API
- POST `/v1/devices`
- POST `/v1/recordings`
- POST `/v1/epochs:ingest`
- POST `/v1/predict`
- GET `/v1/recordings/{id}`
- GET `/v1/recordings/{id}/epochs?from&to`
- GET `/v1/recordings/{id}/predictions?from&to`
- GET `/v1/recordings/{id}/summary?from&to`
- POST `/v1/models/reload`
- GET `/healthz`
- GET `/readyz`
- GET `/metrics`

## Model Registry
Models live under `models/<version>/`:
- `weights.npy`, `bias.npy`
- `label_map.json`
- `feature_schema.json`
- `metadata.json`

Update `ACTIVE_MODEL_VERSION` in `.env` to switch versions. Call `/v1/models/reload` to reload without restart.

## Feature Schema Versioning
`feature_schema.json` defines a version string and feature order. Ingestion and prediction reject mismatches.
The server decodes list, dict, or packed base64 float32 payloads into the canonical float32 vector.

## TimescaleDB Policies
Retention and compression are disabled by default. Enable with:
```
SELECT add_retention_policy('epochs', INTERVAL '90 days');
SELECT add_compression_policy('epochs', INTERVAL '7 days');
```

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
