from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_text(payload: str) -> str:
    return hash_bytes(payload.encode("utf-8"))


def hash_json(payload: object) -> str:
    normalized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hash_text(normalized)


def hash_artifact_dir(
    artifact_dir: Path,
    *,
    exclude_files: Iterable[str] | None = None,
) -> str:
    if not artifact_dir.exists():
        raise FileNotFoundError(artifact_dir)
    excluded = set(exclude_files or [])
    files = sorted(
        path
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.relative_to(artifact_dir).as_posix() not in excluded
    )
    hasher = hashlib.sha256()
    for path in files:
        rel = path.relative_to(artifact_dir).as_posix()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()
