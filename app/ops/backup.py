from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from app.core.settings import get_settings
from app.ops.common import parse_db_url, resolve_compose_command


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

    db = parse_db_url(settings.database_url)

    project_root = Path(__file__).resolve().parents[2]
    compose_command = resolve_compose_command(project_root)

    if compose_command:
        exec_command = compose_command + ["exec", "-T"]
        if db.password:
            exec_command += ["-e", f"PGPASSWORD={db.password}"]
        command = exec_command + [
            "timescaledb",
            "pg_dump",
            "-h",
            db.host,
            "-p",
            str(db.port),
            "-U",
            db.user,
            "-F",
            "c",
            db.database,
        ]
        with output_path.open("wb") as handle:
            subprocess.run(command, check=True, stdout=handle)
    else:
        env = os.environ.copy()
        if db.password:
            env["PGPASSWORD"] = db.password
        command = [
            "pg_dump",
            "-h",
            db.host,
            "-p",
            str(db.port),
            "-U",
            db.user,
            "-F",
            "c",
            "-f",
            str(output_path),
            db.database,
        ]
        subprocess.run(command, check=True, env=env)
    print(f"Backup created at {output_path}")


if __name__ == "__main__":
    main()
