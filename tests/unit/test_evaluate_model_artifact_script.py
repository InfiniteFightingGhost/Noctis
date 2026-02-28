from __future__ import annotations

import json

import numpy as np

from scripts.evaluate_model_artifact import run


def test_evaluate_model_artifact_script_smoke(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    np.save(
        model_dir / "weights.npy", np.asarray([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
    )
    np.save(model_dir / "bias.npy", np.asarray([0.0, 0.0]))
    (model_dir / "label_map.json").write_text(json.dumps(["A", "B"]))
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "feature_strategy": "flatten",
                "window_size": 2,
                "expected_input_dim": 4,
            }
        )
    )

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    X = np.zeros((6, 2, 2), dtype=np.float32)
    X[1] = 1.0
    X[3] = 1.0
    y = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    split_test = np.asarray([4, 5], dtype=np.int64)
    np.savez(dataset_dir / "dataset.npz", X=X, y=y, split_test=split_test)

    code = run(
        [
            "--model-dir",
            str(model_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-prefix",
            "eval_",
        ]
    )
    assert code == 0
    assert (model_dir / "eval_metrics.json").exists()
    assert (model_dir / "eval_calibration.json").exists()
    assert (model_dir / "eval_night_metrics.json").exists()
    assert (model_dir / "eval_domain_metrics.json").exists()
    assert (model_dir / "eval_robustness_report.json").exists()


def test_evaluate_model_artifact_supports_ensemble_dirs(tmp_path) -> None:
    model_a = tmp_path / "model_a"
    model_b = tmp_path / "model_b"
    model_a.mkdir()
    model_b.mkdir()
    for model_dir, sign in ((model_a, 1.0), (model_b, -1.0)):
        np.save(
            model_dir / "weights.npy",
            np.asarray([[sign, -sign], [sign, -sign], [sign, -sign], [sign, -sign]]),
        )
        np.save(model_dir / "bias.npy", np.asarray([0.0, 0.0]))
        (model_dir / "label_map.json").write_text(json.dumps(["A", "B"]))
        (model_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "feature_strategy": "flatten",
                    "window_size": 2,
                    "expected_input_dim": 4,
                }
            )
        )

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    X = np.zeros((6, 2, 2), dtype=np.float32)
    X[1] = 1.0
    y = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    split_test = np.asarray([4, 5], dtype=np.int64)
    np.savez(dataset_dir / "dataset.npz", X=X, y=y, split_test=split_test)

    code = run(
        [
            "--model-dir",
            str(model_a),
            "--ensemble-model-dir",
            str(model_b),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-prefix",
            "ens_",
        ]
    )
    assert code == 0
    assert (model_a / "ens_metrics.json").exists()
