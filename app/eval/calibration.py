# app/eval/calibration.py
from __future__ import annotations
from typing import Any
import numpy as np

from .metrics import CLASSES


def _to_numpy(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def brier_multiclass(y_true: np.ndarray, proba: np.ndarray, num_classes: int) -> float:
    y_true = y_true.astype(np.int64)
    onehot = np.zeros((y_true.shape[0], num_classes), dtype=np.float64)
    onehot[np.arange(y_true.shape[0]), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> dict[str, Any]:
    # conf in [0,1], correct in {0,1}
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)

    out = []
    N = conf.shape[0]
    for b in range(n_bins):
        m = ids == b
        cnt = int(np.sum(m))
        if cnt == 0:
            out.append(
                {
                    "bin": b,
                    "bin_lower": float(bins[b]),
                    "bin_upper": float(bins[b + 1]),
                    "count": 0,
                    "avg_conf": 0.0,
                    "avg_acc": 0.0,
                    "gap": 0.0,
                }
            )
            continue
        avg_conf = float(np.mean(conf[m]))
        avg_acc = float(np.mean(correct[m]))
        out.append(
            {
                "bin": b,
                "bin_lower": float(bins[b]),
                "bin_upper": float(bins[b + 1]),
                "count": cnt,
                "avg_conf": avg_conf,
                "avg_acc": avg_acc,
                "gap": float(abs(avg_acc - avg_conf)),
            }
        )
    return {"n": int(N), "bins": out}


def ece_from_bins(bin_data: dict[str, Any]) -> float:
    N = bin_data["n"]
    if N == 0:
        return 0.0
    ece = 0.0
    for b in bin_data["bins"]:
        if b["count"] == 0:
            continue
        ece += abs(b["avg_acc"] - b["avg_conf"]) * (b["count"] / N)
    return float(ece)


def calibration_metrics(
    y_true,
    proba,
    n_bins: int = 15,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    y_true = _to_numpy(y_true).astype(np.int64)
    proba = _to_numpy(proba).astype(np.float64)
    num_classes = proba.shape[1]
    class_names = class_names or CLASSES[:num_classes]
    if len(class_names) != num_classes:
        raise ValueError("class_names length must match num_classes")

    conf = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)
    correct = (pred == y_true).astype(np.float64)

    bins = reliability_bins(conf, correct, n_bins=n_bins)
    ece = ece_from_bins(bins)
    brier = brier_multiclass(y_true, proba, num_classes)

    # per-class one-vs-rest ECE using proba[:,k] as confidence and y_true==k as label
    per_class = {}
    for k in range(num_classes):
        conf_k = proba[:, k]
        correct_k = (y_true == k).astype(np.float64)
        bins_k = reliability_bins(conf_k, correct_k, n_bins=n_bins)
        per_class[class_names[k]] = {
            "ece": ece_from_bins(bins_k),
            "reliability": bins_k,
        }

    return {
        "brier": brier,
        "ece": ece,
        "class_order": list(class_names),
        "reliability": bins,
        "per_class": per_class,
    }
