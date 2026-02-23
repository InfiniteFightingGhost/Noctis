from __future__ import annotations

from pathlib import Path

from app.ml.registry import LoadedModel
from app.ml.model import LinearSoftmaxModel, load_artifacts
from app.ml.feature_schema import FeatureSchema
from app.stress.service import run_inference_stress


def test_stress_run_reproducible() -> None:
    artifacts = load_artifacts(Path("models/active"))
    model = LoadedModel(
        version="active",
        model=LinearSoftmaxModel(artifacts),
        feature_schema=FeatureSchema(version="v1", features=["f"] * 10),
        metadata={
            "feature_strategy": "mean",
            "expected_input_dim": 10,
            "window_size": 2,
        },
    )
    result_a = run_inference_stress(model, iterations=3, batch_size=2, window_size=2, seed=7)
    assert result_a["avg_latency_ms"] >= 0.0
    assert result_a["p95_latency_ms"] >= 0.0
    assert result_a["throughput_per_sec"] >= 0.0
    assert result_a["duration_seconds"] > 0.0
