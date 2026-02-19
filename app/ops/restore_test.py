from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import psycopg

from app.core.settings import get_settings


@dataclass(frozen=True)
class DbInfo:
    host: str
    port: int
    user: str
    password: str
    database: str


TABLE_CHECKS = {
    "tenants": "id",
    "devices": "id",
    "recordings": "id",
    "epochs": "recording_id || '|' || epoch_start_ts",
    "predictions": "recording_id || '|' || window_end_ts",
}


def _parse_db_url(database_url: str) -> DbInfo:
    url = urlparse(database_url.replace("postgresql+psycopg", "postgresql"))
    return DbInfo(
        host=url.hostname or "localhost",
        port=int(url.port or 5432),
        user=url.username or "postgres",
        password=url.password or "",
        database=(url.path or "/").lstrip("/") or "postgres",
    )


def _resolve_compose_command(project_root: Path) -> list[str] | None:
    compose_file = project_root / "docker-compose.yml"
    if not compose_file.exists():
        return None

    candidates = (["docker", "compose"], ["docker-compose"])
    for base in candidates:
        if shutil.which(base[0]) is None:
            continue
        try:
            subprocess.run(
                base + ["version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        command = base + [
            "-f",
            str(compose_file),
            "--project-directory",
            str(project_root),
        ]
        try:
            result = subprocess.run(
                command + ["ps", "-q", "timescaledb"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError:
            continue
        if result.stdout.strip():
            return command
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate backup restore")
    parser.add_argument("backup_path", help="Path to backup file")
    args = parser.parse_args()

    settings = get_settings()
    backup_path = Path(args.backup_path)
    if not backup_path.exists():
        raise FileNotFoundError(backup_path)

    db = _parse_db_url(settings.database_url)
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
        compose_command = _resolve_compose_command(project_root)

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
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            checksum = conn.execute(
                "SELECT md5(string_agg(md5("
                + key_expr
                + "::text), '' ORDER BY "
                + key_expr
                + ")) FROM "
                + table
            ).fetchone()[0]
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
