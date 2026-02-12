from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from app.core.settings import get_settings


def _parse_db_url(database_url: str) -> dict[str, str]:
    url = urlparse(database_url.replace("postgresql+psycopg", "postgresql"))
    return {
        "host": url.hostname or "localhost",
        "port": str(url.port or 5432),
        "user": url.username or "postgres",
        "password": url.password or "",
        "database": (url.path or "/").lstrip("/") or "postgres",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a logical backup")
    parser.add_argument("--output", help="Override backup output path")
    args = parser.parse_args()

    settings = get_settings()
    backup_dir = Path(settings.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    default_path = backup_dir / f"noctis_backup_{timestamp}.dump"
    output_path = Path(args.output) if args.output else default_path

    db = _parse_db_url(settings.database_url)
    env = os.environ.copy()
    if db["password"]:
        env["PGPASSWORD"] = db["password"]

    command = [
        "pg_dump",
        "-h",
        db["host"],
        "-p",
        db["port"],
        "-U",
        db["user"],
        "-F",
        "c",
        "-f",
        str(output_path),
        db["database"],
    ]
    subprocess.run(command, check=True, env=env)
    print(f"Backup created at {output_path}")


if __name__ == "__main__":
    main()
