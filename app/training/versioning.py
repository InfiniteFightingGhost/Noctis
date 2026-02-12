from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from sqlalchemy.orm import Session

from app.db.models import ModelVersion


SEMVER_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def next_version(
    *,
    session: Session | None,
    output_root: Path,
    bump: str,
) -> str:
    versions = list(_collect_versions(session, output_root))
    if not versions:
        return "0.1.0"
    latest = max(versions, key=_semver_key)
    major, minor, patch = _parse_version(latest)
    if bump == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


def _collect_versions(session: Session | None, output_root: Path) -> Iterable[str]:
    versions: set[str] = set()
    if session is not None:
        rows = session.query(ModelVersion.version).all()
        versions.update(row[0] for row in rows if row and row[0])
    if output_root.exists():
        for child in output_root.iterdir():
            if child.is_dir() and SEMVER_PATTERN.match(child.name):
                versions.add(child.name)
    return versions


def _parse_version(value: str) -> tuple[int, int, int]:
    match = SEMVER_PATTERN.match(value)
    if not match:
        raise ValueError(f"Invalid semantic version: {value}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _semver_key(value: str) -> tuple[int, int, int]:
    return _parse_version(value)
