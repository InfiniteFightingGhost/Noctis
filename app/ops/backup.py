from __future__ import annotations

import argparse
import os
import shutil
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
    parser = argparse.ArgumentParser(description="Create a logical backup")
    parser.add_argument("--output", help="Override backup output path")
    args = parser.parse_args()

    settings = get_settings()
    backup_dir = Path(settings.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    default_path = backup_dir / f"noctis_backup_{timestamp}.dump"
    output_path = Path(args.output) if args.output else default_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db = _parse_db_url(settings.database_url)

    project_root = Path(__file__).resolve().parents[2]
    compose_command = _resolve_compose_command(project_root)

    if compose_command:
        exec_command = compose_command + ["exec", "-T"]
        if db["password"]:
            exec_command += ["-e", f"PGPASSWORD={db['password']}"]
        command = exec_command + [
            "timescaledb",
            "pg_dump",
            "-h",
            db["host"],
            "-p",
            db["port"],
            "-U",
            db["user"],
            "-F",
            "c",
            db["database"],
        ]
        with output_path.open("wb") as handle:
            subprocess.run(command, check=True, stdout=handle)
    else:
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
