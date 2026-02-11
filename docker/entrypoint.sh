#!/usr/bin/env sh
set -e

python - <<'PY'
import os
import time
import psycopg

url = os.getenv("DATABASE_URL")
if not url:
    raise SystemExit("DATABASE_URL not set")

if url.startswith("postgresql+psycopg://"):
    url = url.replace("postgresql+psycopg://", "postgresql://", 1)

for attempt in range(60):
    try:
        with psycopg.connect(url) as conn:
            conn.execute("SELECT 1")
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("Database not reachable")
PY

alembic upgrade head

exec "$@"
