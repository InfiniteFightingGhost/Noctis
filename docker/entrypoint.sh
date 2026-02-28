#!/usr/bin/env sh
set -e

python - <<'PY'
import os
import subprocess
import time
import sys

import psycopg

url = os.getenv("DATABASE_URL", "").strip()
if not url:
    print("ERROR: DATABASE_URL not set", file=sys.stderr)
    raise SystemExit("DATABASE_URL not set")

print(f"[entrypoint] Database URL: {url.split('@')[1] if '@' in url else 'invalid format'}", flush=True)

if url.startswith("postgresql+psycopg://"):
    url = url.replace("postgresql+psycopg://", "postgresql://", 1)

# Extract the database name from the URL (last part after the last /)
url_parts = url.rsplit("/", 1)
db_name = url_parts[1].strip() if len(url_parts) > 1 else "noctis"
postgres_url = url_parts[0] + "/postgres"  # Connect to default postgres DB to create if needed

for attempt in range(60):
    try:
        with psycopg.connect(postgres_url) as conn:
            conn.execute("SELECT 1")
        print(f"[entrypoint] Database server reachable (attempt {attempt + 1})", flush=True)
        break
    except Exception as e:
        if attempt == 0 or attempt == 59:
            print(f"[entrypoint] Database server not reachable attempt={attempt + 1} error={e}", flush=True)
        time.sleep(1)
else:
    print("[entrypoint] Database server never became reachable after 60 attempts", file=sys.stderr, flush=True)
    raise SystemExit("Database not reachable")

# Skip database creation if we're connecting directly to an existing database
if url_parts[0].endswith(f":5432/{db_name}"):
    print(f"[entrypoint] Connected directly to database '{db_name}' - skipping creation check", flush=True)
else:
    # Create the database if it doesn't exist
    print(f"[entrypoint] Creating database '{db_name}' if it doesn't exist...", flush=True)
    try:
        with psycopg.connect(postgres_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
                exists = cur.fetchone()
                if not exists:
                    cur.execute(f"CREATE DATABASE {db_name}")
                    print(f"[entrypoint] Created database '{db_name}'", flush=True)
                else:
                    print(f"[entrypoint] Database '{db_name}' already exists", flush=True)
    except psycopg.errors.DuplicateDatabase:
        print(f"[entrypoint] Database '{db_name}' already exists (DuplicateDatabase error handled)", flush=True)
    except Exception as e:
        print(f"[entrypoint] Error checking/creating database: {e}", file=sys.stderr, flush=True)
        raise SystemExit("Failed to check/create database")

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
        print("[entrypoint] Running alembic migrations...", flush=True)
        subprocess.run(["alembic", "upgrade", "head"], check=True)
        print("[entrypoint] Alembic migrations completed", flush=True)
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
PY

if [ -n "$PROMETHEUS_MULTIPROC_DIR" ]; then
  mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
  rm -f "$PROMETHEUS_MULTIPROC_DIR"/*
fi

exec "$@"
