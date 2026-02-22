#!/usr/bin/env sh
set -e

python - <<'PY'
import os
import subprocess
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

lock_id = 9423501

with psycopg.connect(url) as conn:
    conn.execute("SELECT pg_advisory_lock(%s)", (lock_id,))
    try:
        conn.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM pg_type t
                    JOIN pg_namespace n ON n.oid = t.typnamespace
                    WHERE t.typname = 'alembic_version' AND n.nspname = 'public'
                ) AND NOT EXISTS (
                    SELECT 1
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname = 'alembic_version' AND c.relkind = 'r' AND n.nspname = 'public'
                ) THEN
                    EXECUTE 'DROP TYPE public.alembic_version';
                END IF;
            END $$;
            """
        )
        subprocess.run(["alembic", "upgrade", "head"], check=True)
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
PY

exec "$@"
