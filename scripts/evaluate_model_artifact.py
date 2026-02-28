from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.eval import evaluate_all
from app.ml.model import load_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model artifact on dataset split")
    parser.add_argument("--model-dir", required=True, help="Path to model artifact directory")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset snapshot directory")
    parser.add_argument("--split", required=True, choices=["test", "val"], help="Split to evaluate")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional filename prefix for output files",
    )
    parser.add_argument(
        "--ensemble-model-dir",
        action="append",
        default=[],
        help="Optional additional model directory for logit/probability averaging",
    )
    return parser.parse_args(argv)


def _extract_split_indices(dataset: dict[str, Any], split: str) -> np.ndarray:
    values = dataset.get(f"split_{split}")
    if values is None:
        raise ValueError(f"Dataset split_{split} is missing")
    indices = np.asarray(values, dtype=np.int64)
    if indices.size == 0:
        raise ValueError(f"Dataset split_{split} is empty")
    return indices


def _prepare_inputs(
    dataset: dict[str, Any], indices: np.ndarray, feature_strategy: str
) -> np.ndarray:
    X = np.asarray(dataset["X"])
    batch = X[indices]
    if feature_strategy == "sequence":
        if batch.ndim != 3:
            raise ValueError("Sequence model expects 3D X")
        return batch.astype(np.float32)
    if feature_strategy == "mean":
        if batch.ndim == 3:
            return batch.mean(axis=1).astype(np.float32)
        if batch.ndim == 2:
            return batch.astype(np.float32)
        raise ValueError("Mean feature strategy expects 2D or 3D X")
    if feature_strategy == "flatten":
        return batch.reshape(batch.shape[0], -1).astype(np.float32)
    raise ValueError(f"Unsupported feature strategy: {feature_strategy}")


def _resolve_dataset_ids(dataset: dict[str, Any], indices: np.ndarray) -> np.ndarray:
    values = dataset.get("dataset_ids")
    if values is None:
        return np.full(indices.shape[0], "UNKNOWN", dtype=object)
    dataset_ids = np.asarray(values).astype(str)
    return dataset_ids[indices]


def _resolve_recording_ids(dataset: dict[str, Any], indices: np.ndarray) -> np.ndarray:
    values = dataset.get("recording_ids")
    if values is None:
        return np.asarray([f"recording_{idx}" for idx in indices], dtype=object)
    recording_ids = np.asarray(values).astype(str)
    return recording_ids[indices]


def _resolve_labels(
    bundle_labels: list[str], dataset: dict[str, Any], y_eval: np.ndarray
) -> list[str]:
    labels = [str(label) for label in bundle_labels]
    if labels:
        return labels
    label_map = dataset.get("label_map")
    if label_map is not None:
        return [str(label) for label in np.asarray(label_map).tolist()]
    uniques = sorted({str(v) for v in y_eval.tolist()})
    return uniques


def _encode_labels(y_eval: np.ndarray, labels: list[str]) -> np.ndarray:
    if y_eval.dtype.kind in {"i", "u"}:
        y_idx = y_eval.astype(np.int64)
        if y_idx.size > 0 and int(np.max(y_idx)) < len(labels):
            return y_idx
    lookup = {label: idx for idx, label in enumerate(labels)}
    encoded = np.asarray([lookup[str(value)] for value in y_eval], dtype=np.int64)
    return encoded


def _predict_probabilities(
    *,
    bundle,
    batch: np.ndarray,
    dataset_ids: np.ndarray,
    feature_strategy: str,
    batch_size: int = 256,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, batch.shape[0], batch_size):
        end = min(start + batch_size, batch.shape[0])
        if feature_strategy == "sequence":
            probs = bundle.model.predict_proba(batch[start:end], dataset_ids=dataset_ids[start:end])
        else:
            probs = bundle.model.predict_proba(batch[start:end])
        rows.append(probs)
    return np.concatenate(rows, axis=0)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_dir = Path(args.model_dir)
    dataset_dir = Path(args.dataset_dir)
    dataset = dict(np.load(dataset_dir / "dataset.npz", allow_pickle=True))
    indices = _extract_split_indices(dataset, args.split)

    bundle = load_model(model_dir)
    feature_strategy = str(bundle.metadata.get("feature_strategy") or "mean")
    batch = _prepare_inputs(dataset, indices, feature_strategy)
    dataset_ids = _resolve_dataset_ids(dataset, indices)
    recording_ids = _resolve_recording_ids(dataset, indices)
    y_eval = np.asarray(dataset["y"])[indices]
    labels = _resolve_labels(bundle.model.labels, dataset, y_eval)
    y_true = _encode_labels(y_eval, labels)

    probs = _predict_probabilities(
        bundle=bundle,
        batch=batch,
        dataset_ids=dataset_ids,
        feature_strategy=feature_strategy,
    )
    if args.ensemble_model_dir:
        ensemble_probs = [probs]
        for model_path in args.ensemble_model_dir:
            other_bundle = load_model(Path(model_path))
            other_feature_strategy = str(other_bundle.metadata.get("feature_strategy") or "mean")
            other_probs = _predict_probabilities(
                bundle=other_bundle,
                batch=batch,
                dataset_ids=dataset_ids,
                feature_strategy=other_feature_strategy,
            )
            if other_probs.shape != probs.shape:
                raise ValueError("Ensemble models must output same probability shape")
            ensemble_probs.append(other_probs)
        probs = np.mean(np.stack(ensemble_probs, axis=0), axis=0)
    y_pred = np.argmax(probs, axis=1)
    scorecard = evaluate_all(
        y_true=y_true,
        y_pred=y_pred,
        proba=probs,
        recording_id=recording_ids,
        dataset_id=dataset_ids,
        class_names=labels,
    )

    prefix = str(args.output_prefix or "")
    _write_json(model_dir / f"{prefix}metrics.json", scorecard)
    _write_json(model_dir / f"{prefix}calibration.json", scorecard["calibration"])
    _write_json(model_dir / f"{prefix}night_metrics.json", scorecard["night"])
    _write_json(model_dir / f"{prefix}domain_metrics.json", scorecard["domain"])
    _write_json(model_dir / f"{prefix}robustness_report.json", scorecard["robustness"])
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
