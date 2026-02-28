# app/eval/domain_analysis.py
from __future__ import annotations
from typing import Any
import numpy as np

from .metrics import CLASSES, evaluate_classification


def _to_numpy(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def per_dataset_eval(
    y_true,
    y_pred,
    proba,
    dataset_id,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    y_true = _to_numpy(y_true).astype(np.int64)
    y_pred = _to_numpy(y_pred).astype(np.int64)
    proba = _to_numpy(proba).astype(np.float64)
    dataset_id = _to_numpy(dataset_id)

    num_classes = proba.shape[1]
    class_names = class_names or CLASSES[:num_classes]
    out = {}
    for ds in sorted(np.unique(dataset_id).tolist(), key=lambda value: str(value)):
        m = dataset_id == ds
        metrics = evaluate_classification(
            y_true[m],
            y_pred[m],
            num_classes=num_classes,
            class_names=class_names,
        )
        metrics["samples"] = int(m.sum())
        out[str(ds)] = metrics
    return out


def domain_variance(per_ds: dict[str, Any]) -> dict[str, float]:
    if not per_ds:
        return {"macro_f1": 0.0, "mcc": 0.0, "kappa": 0.0}

    def spread(metric: str) -> float:
        values = [float(payload.get(metric, 0.0)) for payload in per_ds.values()]
        return float(max(values) - min(values)) if values else 0.0

    return {
        "macro_f1": spread("macro_f1"),
        "mcc": spread("mcc"),
        "kappa": spread("kappa"),
    }
