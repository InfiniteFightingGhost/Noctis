# app/eval/metrics.py
from __future__ import annotations
from typing import Any
import numpy as np

CLASSES = ["W", "Light", "Deep", "REM"]


def _to_numpy(a) -> np.ndarray:
    if a is None:
        return np.asarray([])
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> dict[str, Any]:
    # rows=true, cols=pred
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.int64)
    tn = cm.sum().astype(np.float64) - (tp + fp + fn)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tp), where=(tn + fp) > 0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "support": support,
    }


def macro_f1(per_class_f1: np.ndarray, support: np.ndarray) -> float:
    # macro averages over classes with support>0
    mask = support > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(per_class_f1[mask]))


def weighted_f1(per_class_f1: np.ndarray, support: np.ndarray) -> float:
    tot = support.sum()
    if tot == 0:
        return 0.0
    return float(np.sum(per_class_f1 * support) / tot)


def balanced_accuracy(per_class_recall: np.ndarray, support: np.ndarray) -> float:
    mask = support > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(per_class_recall[mask]))


def mcc_multiclass(cm: np.ndarray) -> float:
    # Gorodkin (2004) multiclass MCC implementation via confusion matrix
    t_sum = cm.sum(axis=1).astype(np.float64)
    p_sum = cm.sum(axis=0).astype(np.float64)
    n = cm.sum().astype(np.float64)
    c = np.trace(cm).astype(np.float64)

    s = 0.0
    for k in range(cm.shape[0]):
        for l in range(cm.shape[0]):
            for m in range(cm.shape[0]):
                s += cm[k, k] * cm[l, m] - cm[k, l] * cm[m, k]

    denom = np.sqrt((n**2 - np.sum(p_sum**2)) * (n**2 - np.sum(t_sum**2)))
    if denom == 0:
        return 0.0
    # Equivalent simpler form:
    num = c * n - np.dot(t_sum, p_sum)
    return float(num / denom)


def cohen_kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    row = cm.sum(axis=1).astype(np.float64)
    col = cm.sum(axis=0).astype(np.float64)
    pe = np.dot(row, col) / (n * n)
    denom = 1.0 - pe
    if denom == 0:
        return 0.0
    return float((po - pe) / denom)


def evaluate_classification(
    y_true,
    y_pred,
    num_classes: int = 4,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    y_true = _to_numpy(y_true).astype(np.int64)
    y_pred = _to_numpy(y_pred).astype(np.int64)
    class_names = class_names or CLASSES[:num_classes]
    if len(class_names) != num_classes:
        raise ValueError("class_names length must match num_classes")

    cm = confusion_matrix(y_true, y_pred, num_classes)
    prf = per_class_prf(cm)

    out = {
        "confusion_matrix": cm.tolist(),
        "class_order": list(class_names),
        "per_class": {
            class_names[i]: {
                "precision": float(prf["precision"][i]),
                "recall": float(prf["recall"][i]),
                "f1": float(prf["f1"][i]),
                "specificity": float(prf["specificity"][i]),
                "support": int(prf["support"][i]),
            }
            for i in range(num_classes)
        },
        "macro_f1": macro_f1(prf["f1"], prf["support"]),
        "weighted_f1": weighted_f1(prf["f1"], prf["support"]),
        "balanced_accuracy": balanced_accuracy(prf["recall"], prf["support"]),
        "mcc": mcc_multiclass(cm),
        "kappa": cohen_kappa(cm),
        "support_total": int(prf["support"].sum()),
    }
    return out


def slice_by_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return arr[mask]


def evaluate_slices(y_true, y_pred, proba, recording_id, dataset_id) -> dict[str, Any]:
    y_true = _to_numpy(y_true).astype(np.int64)
    y_pred = _to_numpy(y_pred).astype(np.int64)
    proba = _to_numpy(proba).astype(np.float64)
    recording_id = _to_numpy(recording_id)
    dataset_id = _to_numpy(dataset_id)

    return {
        "global": evaluate_classification(y_true, y_pred, num_classes=proba.shape[1]),
    }
