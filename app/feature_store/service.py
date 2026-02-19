from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import FeatureSchema, FeatureSchemaFeature
from app.feature_store.schema import (
    FeatureDefinition,
    FeatureSchemaRecord,
    feature_schema_hash,
    parse_feature_schema_payload,
)


def list_feature_schemas(session: Session) -> list[FeatureSchemaRecord]:
    rows = (
        session.execute(select(FeatureSchema).order_by(FeatureSchema.created_at))
        .scalars()
        .all()
    )
    return [_schema_record(row) for row in rows]


def get_feature_schema_by_version(
    session: Session, version: str
) -> FeatureSchemaRecord | None:
    row = session.execute(
        select(FeatureSchema).where(FeatureSchema.version == version)
    ).scalar_one_or_none()
    if row is None:
        return None
    return _schema_record(row)


def get_active_feature_schema(
    session: Session, *, allow_missing: bool = False
) -> FeatureSchemaRecord | None:
    row = session.execute(
        select(FeatureSchema).where(FeatureSchema.is_active.is_(True))
    ).scalar_one_or_none()
    if row is None:
        if allow_missing:
            return None
        raise ValueError("Active feature schema not found")
    return _schema_record(row)


def register_feature_schema(
    session: Session,
    *,
    payload: dict[str, Any],
    activate: bool = False,
) -> FeatureSchemaRecord:
    version, description, features = parse_feature_schema_payload(payload)
    existing = get_feature_schema_by_version(session, version)
    schema_hash = feature_schema_hash(features)
    if existing:
        if existing.hash != schema_hash:
            raise ValueError(
                "Feature schema version already exists with different hash"
            )
        if activate and not existing.is_active:
            _activate_schema(session, existing.id)
        return existing
    active = get_active_feature_schema(session, allow_missing=True)
    if active:
        _validate_schema_evolution(active, features)
    schema = FeatureSchema(
        version=version,
        hash=schema_hash,
        description=description,
        is_active=activate,
        created_at=datetime.now(timezone.utc),
    )
    session.add(schema)
    session.flush()
    for feature in features:
        session.add(
            FeatureSchemaFeature(
                feature_schema_id=schema.id,
                name=feature.name,
                dtype=feature.dtype,
                allowed_range=feature.allowed_range,
                description=feature.description,
                introduced_in_version=feature.introduced_in_version,
                deprecated_in_version=feature.deprecated_in_version,
                position=feature.position,
            )
        )
    if activate:
        _activate_schema(session, schema.id)
    session.flush()
    return _schema_record(schema)


def ensure_active_schema_from_path(
    session: Session,
    *,
    schema_path: Path,
    activate: bool = True,
) -> FeatureSchemaRecord:
    active = get_active_feature_schema(session, allow_missing=True)
    if active:
        return active
    if not schema_path.exists():
        raise ValueError("Feature schema path not found")
    payload = json.loads(schema_path.read_text())
    return register_feature_schema(session, payload=payload, activate=activate)


def _schema_record(schema: FeatureSchema) -> FeatureSchemaRecord:
    features = [
        FeatureDefinition(
            name=feature.name,
            dtype=feature.dtype,
            allowed_range=feature.allowed_range,
            description=feature.description,
            introduced_in_version=feature.introduced_in_version or schema.version,
            deprecated_in_version=feature.deprecated_in_version,
            position=feature.position,
        )
        for feature in sorted(schema.features, key=lambda item: item.position)
    ]
    return FeatureSchemaRecord(
        id=schema.id,
        version=schema.version,
        hash=schema.hash,
        description=schema.description,
        is_active=schema.is_active,
        created_at=schema.created_at,
        features=features,
    )


def _activate_schema(session: Session, schema_id) -> None:
    session.query(FeatureSchema).filter(FeatureSchema.is_active.is_(True)).update(
        {FeatureSchema.is_active: False}
    )
    session.query(FeatureSchema).filter(FeatureSchema.id == schema_id).update(
        {FeatureSchema.is_active: True}
    )


def _validate_schema_evolution(
    active_schema: FeatureSchemaRecord, new_features: list[FeatureDefinition]
) -> None:
    existing_names = set(active_schema.feature_names)
    new_names = {feature.name for feature in new_features}
    removed = existing_names - new_names
    if not removed:
        return
    lookup = active_schema.feature_lookup()
    for name in removed:
        definition = lookup.get(name)
        if definition is None or not definition.deprecated_in_version:
            raise ValueError(
                "Feature removal requires a deprecation phase in prior schema"
            )
