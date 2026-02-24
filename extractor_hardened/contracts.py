from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Contracts:
    schema_manifest: dict[str, Any]
    alignment_policy: dict[str, Any]
    qc_policy: dict[str, Any]
    feature_manifest: dict[str, Any]
    schema_hash: str


def load_contracts(root: Path | None = None) -> Contracts:
    base = root or REPO_ROOT
    schema_path = base / "schema_manifest.json"
    alignment_path = base / "alignment_policy.yaml"
    qc_path = base / "qc_policy.yaml"
    feature_path = base / "feature_manifest.yaml"

    schema_manifest = json.loads(schema_path.read_text(encoding="utf-8"))
    alignment_policy = yaml.safe_load(alignment_path.read_text(encoding="utf-8"))
    qc_policy = yaml.safe_load(qc_path.read_text(encoding="utf-8"))
    feature_manifest = yaml.safe_load(feature_path.read_text(encoding="utf-8"))

    schema_hash = hashlib.sha256(
        json.dumps(schema_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return Contracts(
        schema_manifest=schema_manifest,
        alignment_policy=alignment_policy,
        qc_policy=qc_policy,
        feature_manifest=feature_manifest,
        schema_hash=schema_hash,
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
