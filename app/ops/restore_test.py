from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from app.core.settings import get_settings
from app.ops.common import DbInfo, parse_db_url, resolve_compose_command


TABLE_CHECKS = {
    "tenants": "id",
    "devices": "id",
    "recordings": "id",
    "epochs": "recording_id || '|' || epoch_start_ts",
    "predictions": "recording_id || '|' || window_end_ts",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate backup restore")
    parser.add_argument("backup_path", help="Path to backup file")
    args = parser.parse_args()

    settings = get_settings()
    backup_path = Path(args.backup_path)
    if not backup_path.exists():
        raise FileNotFoundError(backup_path)

    db = parse_db_url(settings.database_url)
    temp_db = f"{db.database}_restore_test_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    with psycopg.connect(
        host=db.host,
        port=db.port,
        user=db.user,
        password=db.password,
        dbname="postgres",
        autocommit=True,
    ) as conn:
        conn.execute(f"CREATE DATABASE {temp_db}")

    try:
        project_root = Path(__file__).resolve().parents[2]
        compose_command = resolve_compose_command(project_root)

        if compose_command:
            exec_command = compose_command + ["exec", "-T"]
            if db.password:
                exec_command += ["-e", f"PGPASSWORD={db.password}"]
            exec_command += ["-e", "PGOPTIONS=-c timescaledb.restoring=on"]
            restore_cmd = exec_command + [
                "timescaledb",
                "pg_restore",
                "-h",
                db.host,
                "-p",
                str(db.port),
                "-U",
                db.user,
                "-d",
                temp_db,
            ]
            with backup_path.open("rb") as handle:
                subprocess.run(restore_cmd, check=True, stdin=handle)
        else:
            env = os.environ.copy()
            if db.password:
                env["PGPASSWORD"] = db.password
            env["PGOPTIONS"] = "-c timescaledb.restoring=on"
            restore_cmd = [
                "pg_restore",
                "-h",
                db.host,
                "-p",
                str(db.port),
                "-U",
                db.user,
                "-d",
                temp_db,
                str(backup_path),
            ]
            subprocess.run(restore_cmd, check=True, env=env)

        original_stats = _collect_stats(db)
        restored_stats = _collect_stats(
            DbInfo(
                host=db.host,
                port=db.port,
                user=db.user,
                password=db.password,
                database=temp_db,
            )
        )

        _compare_stats(original_stats, restored_stats)
        print("Restore validation succeeded")
    finally:
        with psycopg.connect(
            host=db.host,
            port=db.port,
            user=db.user,
            password=db.password,
            dbname="postgres",
            autocommit=True,
        ) as conn:
            conn.execute(f"DROP DATABASE IF EXISTS {temp_db}")


def _collect_stats(db: DbInfo) -> dict[str, dict[str, str]]:
    stats: dict[str, dict[str, str]] = {}
    with psycopg.connect(
        host=db.host,
        port=db.port,
        user=db.user,
        password=db.password,
        dbname=db.database,
    ) as conn:
        for table, key_expr in TABLE_CHECKS.items():
            count_row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            if count_row is None:
                raise ValueError(f"Missing count for table {table}")
            count = count_row[0]
            checksum_row = conn.execute(
                "SELECT md5(string_agg(md5("
                + key_expr
                + "::text), '' ORDER BY "
                + key_expr
                + ")) FROM "
                + table
            ).fetchone()
            if checksum_row is None:
                raise ValueError(f"Missing checksum for table {table}")
            checksum = checksum_row[0]
            stats[table] = {"count": str(count), "checksum": str(checksum)}
    return stats


def _compare_stats(
    original: dict[str, dict[str, str]],
    restored: dict[str, dict[str, str]],
) -> None:
    for table, original_stats in original.items():
        restored_stats = restored.get(table)
        if restored_stats is None:
            raise ValueError(f"Missing table in restored DB: {table}")
        if original_stats["count"] != restored_stats["count"]:
            raise ValueError(
                f"Row count mismatch for {table}: {original_stats['count']} != {restored_stats['count']}"
            )
        if original_stats["checksum"] != restored_stats["checksum"]:
            raise ValueError(
                f"Checksum mismatch for {table}: {original_stats['checksum']} != {restored_stats['checksum']}"
            )


if __name__ == "__main__":
    main()
