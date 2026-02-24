from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class DbInfo:
    host: str
    port: int
    user: str
    password: str
    database: str


def parse_db_url(database_url: str) -> DbInfo:
    url = urlparse(database_url.replace("postgresql+psycopg", "postgresql"))
    return DbInfo(
        host=url.hostname or "localhost",
        port=int(url.port or 5432),
        user=url.username or "postgres",
        password=url.password or "",
        database=(url.path or "/").lstrip("/") or "postgres",
    )


def resolve_compose_command(project_root: Path) -> list[str] | None:
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
