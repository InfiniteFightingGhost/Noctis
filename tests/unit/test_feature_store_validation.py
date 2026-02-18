from __future__ import annotations

import base64
from datetime import datetime, timezone
import uuid

import numpy as np
import pytest

from app.feature_store.schema import FeatureDefinition, FeatureSchemaRecord
from app.feature_store.validation import validate_feature_payload


def _schema() -> FeatureSchemaRecord:
    return FeatureSchemaRecord(
        id=uuid.uuid4(),
        version="v1",
        hash="hash",
        description=None,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        features=[
            FeatureDefinition(
                name="f1",
                dtype="float32",
                allowed_range={"min": 0.0, "max": 1.0},
                description=None,
                introduced_in_version="v1",
                deprecated_in_version=None,
                position=0,
            ),
            FeatureDefinition(
                name="f2",
                dtype="float32",
                allowed_range=None,
                description=None,
                introduced_in_version="v1",
                deprecated_in_version=None,
                position=1,
            ),
        ],
    )


def test_validate_feature_payload_list() -> None:
    schema = _schema()
    result = validate_feature_payload([0.5, 2.0], schema)
    assert result == [0.5, 2.0]


def test_validate_feature_payload_missing_feature() -> None:
    schema = _schema()
    with pytest.raises(ValueError):
        validate_feature_payload({"f1": 0.5}, schema)


def test_validate_feature_payload_extra_feature() -> None:
    schema = _schema()
    with pytest.raises(ValueError):
        validate_feature_payload({"f1": 0.5, "f2": 0.1, "f3": 0.2}, schema)


def test_validate_feature_payload_range_violation() -> None:
    schema = _schema()
    with pytest.raises(ValueError):
        validate_feature_payload({"f1": 2.0, "f2": 0.1}, schema)


def test_validate_feature_payload_base64() -> None:
    schema = _schema()
    payload = base64.b64encode(np.array([0.2, 0.3], dtype=np.float32).tobytes()).decode(
        "utf-8"
    )
    result = validate_feature_payload(payload, schema)
    assert result == payload
