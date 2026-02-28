from __future__ import annotations

from typing import Any

import numpy as np

from .calibration import calibration_metrics
from .metrics import CLASSES, evaluate_classification

FORBIDDEN_DEFAULT = {(2, 3), (3, 2)}


def _to_numpy(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def levenshtein(a: list[int], b: list[int]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[n, m])


def transition_matrix(seq: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(max(len(seq) - 1, 0)):
        matrix[int(seq[i]), int(seq[i + 1])] += 1
    return matrix


def row_normalize(mat: np.ndarray) -> np.ndarray:
    sums = mat.sum(axis=1, keepdims=True).astype(np.float64)
    return np.divide(mat, sums, out=np.zeros_like(mat, dtype=np.float64), where=sums > 0)


def entropy_rows(mat_prob: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(mat_prob, eps, 1.0)
    return float(np.mean(-np.sum(p * np.log(p), axis=1)))


def runs_count(seq: np.ndarray) -> int:
    if len(seq) == 0:
        return 0
    return int(1 + np.sum(seq[1:] != seq[:-1]))


def derive_clinical(seq: np.ndarray, epoch_seconds: int) -> dict[str, float]:
    n = len(seq)
    if n == 0:
        return {
            "tst_min": 0.0,
            "se": 0.0,
            "rem_pct": 0.0,
            "n3_pct": 0.0,
            "rem_latency_min": 0.0,
            "waso_min": 0.0,
        }

    is_wake = seq == 0
    tib_min = (n * epoch_seconds) / 60.0
    sleep_epochs = int(np.sum(~is_wake))
    tst_min = (sleep_epochs * epoch_seconds) / 60.0
    se = float(tst_min / tib_min) if tib_min > 0 else 0.0

    sleep_onset_idx = int(np.argmax(~is_wake)) if sleep_epochs > 0 else None

    rem_epochs = int(np.sum(seq == 3))
    n3_epochs = int(np.sum(seq == 2))
    rem_pct = float(rem_epochs / sleep_epochs) if sleep_epochs > 0 else 0.0
    n3_pct = float(n3_epochs / sleep_epochs) if sleep_epochs > 0 else 0.0

    rem_latency_min = 0.0
    if sleep_onset_idx is not None:
        rem_after = np.where(seq[sleep_onset_idx:] == 3)[0]
        if rem_after.size > 0:
            rem_latency_min = float((rem_after[0] * epoch_seconds) / 60.0)

    waso_min = 0.0
    if sleep_onset_idx is not None:
        waso_epochs = int(np.sum(seq[sleep_onset_idx:] == 0))
        waso_min = float((waso_epochs * epoch_seconds) / 60.0)

    return {
        "tst_min": float(tst_min),
        "se": float(se),
        "rem_pct": float(rem_pct),
        "n3_pct": float(n3_pct),
        "rem_latency_min": float(rem_latency_min),
        "waso_min": float(waso_min),
    }


def _clinical_error(
    true_values: dict[str, float], pred_values: dict[str, float]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in sorted(true_values.keys()):
        abs_err = float(abs(pred_values[key] - true_values[key]))
        denom = abs(true_values[key])
        ape = 0.0 if denom <= 1e-6 else float(abs_err / denom)
        out[key] = {"mae": abs_err, "ape": ape}
    return out


def per_night_eval(
    y_true,
    y_pred,
    proba,
    recording_id,
    dataset_id,
    *,
    class_names: list[str] | None = None,
    epoch_seconds: int = 30,
    calibration_bins: int = 15,
    forbidden_transitions: set[tuple[int, int]] = FORBIDDEN_DEFAULT,
) -> dict[str, Any]:
    y_true = _to_numpy(y_true).astype(np.int64)
    y_pred = _to_numpy(y_pred).astype(np.int64)
    proba = _to_numpy(proba).astype(np.float64)
    recording_id = _to_numpy(recording_id)
    dataset_id = _to_numpy(dataset_id)

    num_classes = proba.shape[1]
    class_names = class_names or CLASSES[:num_classes]
    nights = sorted(np.unique(recording_id).tolist(), key=lambda value: str(value))

    night_rows = []
    for rid in nights:
        mask = recording_id == rid
        yt = y_true[mask]
        yp = y_pred[mask]
        pr = proba[mask]
        ds = str(sorted(np.unique(dataset_id[mask]).tolist(), key=lambda value: str(value))[0])

        cls = evaluate_classification(yt, yp, num_classes=num_classes, class_names=class_names)
        cal = calibration_metrics(yt, pr, n_bins=calibration_bins, class_names=class_names)

        transitions_true = transition_matrix(yt, num_classes)
        transitions_pred = transition_matrix(yp, num_classes)
        transitions_true_norm = row_normalize(transitions_true)
        transitions_pred_norm = row_normalize(transitions_pred)

        transition_matrix_error_raw = float(
            np.abs(transitions_pred.astype(np.float64) - transitions_true.astype(np.float64)).sum()
        )
        transition_matrix_error_norm = float(
            np.abs(transitions_pred_norm - transitions_true_norm).sum()
        )
        transition_entropy_diff = abs(
            entropy_rows(transitions_pred_norm) - entropy_rows(transitions_true_norm)
        )

        edit_distance_raw = levenshtein(yt.tolist(), yp.tolist())
        edit_distance_norm = float(edit_distance_raw / max(len(yt), len(yp), 1))

        clinical_true = derive_clinical(yt, epoch_seconds)
        clinical_pred = derive_clinical(yp, epoch_seconds)
        eps = 1e-6
        frag_true = (runs_count(yt) - 1) / (clinical_true["tst_min"] + eps)
        frag_pred = (runs_count(yp) - 1) / (clinical_pred["tst_min"] + eps)
        fragmentation_error = float(abs(frag_pred - frag_true))

        forbidden = 0
        total_transitions = max(len(yp) - 1, 0)
        for i in range(total_transitions):
            if (int(yp[i]), int(yp[i + 1])) in forbidden_transitions:
                forbidden += 1
        impossible_transition_rate = (
            float(forbidden / total_transitions) if total_transitions > 0 else 0.0
        )

        night_rows.append(
            {
                "recording_id": str(rid),
                "dataset_id": ds,
                "samples": int(mask.sum()),
                "macro_f1": cls["macro_f1"],
                "mcc": cls["mcc"],
                "kappa": cls["kappa"],
                "brier": cal["brier"],
                "ece": cal["ece"],
                "transition_matrix_error_raw": transition_matrix_error_raw,
                "transition_matrix_error_norm": transition_matrix_error_norm,
                "transition_entropy_diff": float(transition_entropy_diff),
                "hypnogram_edit_distance_raw": int(edit_distance_raw),
                "hypnogram_edit_distance_norm": edit_distance_norm,
                "fragmentation_error": fragmentation_error,
                "impossible_transition_rate": impossible_transition_rate,
                "wake_deep_direct_rate": _transition_pair_rate(yp, src=0, dst=2),
                "clinical_true": clinical_true,
                "clinical_pred": clinical_pred,
                "clinical_error": _clinical_error(clinical_true, clinical_pred),
            }
        )

    return {"class_order": class_names, "nights": night_rows}


def _transition_pair_rate(seq: np.ndarray, *, src: int, dst: int) -> float:
    total_transitions = max(len(seq) - 1, 0)
    if total_transitions == 0:
        return 0.0
    count = 0
    for idx in range(total_transitions):
        if int(seq[idx]) == src and int(seq[idx + 1]) == dst:
            count += 1
    return float(count / total_transitions)
