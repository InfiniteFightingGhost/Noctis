from __future__ import annotations

import base64

import numpy as np

from app.ml.feature_decode import decode_features
from app.ml.feature_schema import FeatureSchema


def test_decode_features_list() -> None:
    schema = FeatureSchema(version="v1", features=["a", "b", "c"])
    vector = decode_features([1, 2, 3], schema)
    assert vector.dtype == np.float32
    assert vector.tolist() == [1.0, 2.0, 3.0]


def test_decode_features_dict() -> None:
    schema = FeatureSchema(version="v1", features=["a", "b", "c"])
    vector = decode_features({"a": 1, "b": 2, "c": 3}, schema)
    assert vector.tolist() == [1.0, 2.0, 3.0]


def test_decode_features_base64() -> None:
    schema = FeatureSchema(version="v1", features=["a", "b", "c"])
    raw = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
    payload = base64.b64encode(raw).decode("utf-8")
    vector = decode_features(payload, schema)
    assert vector.tolist() == [1.0, 2.0, 3.0]
