from __future__ import annotations

import pytest

from app.schemas.epochs import EpochIngest


def test_epoch_ingest_rejects_invalid_features() -> None:
    with pytest.raises(ValueError):
        EpochIngest(
            epoch_index=0,
            epoch_start_ts="2026-02-01T00:00:00Z",
            feature_schema_version="v1",
            features=123,
        )
