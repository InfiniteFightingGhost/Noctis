from __future__ import annotations

from collections import Counter
from math import log


def build_labels(y_true: list[str], y_pred: list[str]) -> list[str]:
    labels = sorted(set(y_true) | set(y_pred))
    return labels


def confusion_matrix(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for truth, pred in zip(y_true, y_pred, strict=True):
        matrix[index[truth]][index[pred]] += 1
    return matrix


def per_class_metrics(
    matrix: list[list[int]], labels: list[str]
) -> list[dict[str, object]]:
    metrics: list[dict[str, object]] = []
    size = len(labels)
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[row][i] for row in range(size)) - tp
        fn = sum(matrix[i][col] for col in range(size)) - tp
        support = sum(matrix[i])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        metrics.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return metrics


def accuracy(matrix: list[list[int]]) -> float:
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    total = sum(sum(row) for row in matrix)
    return correct / total if total > 0 else 0.0


def macro_f1(metrics: list[dict[str, object]]) -> float:
    if not metrics:
        return 0.0
    return sum(float(item["f1"]) for item in metrics) / len(metrics)


def transition_matrix(stages: list[str], labels: list[str]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for prev, curr in zip(stages, stages[1:], strict=False):
        if prev in index and curr in index:
            matrix[index[prev]][index[curr]] += 1
    return matrix


def confidence_histogram(confidences: list[float], bins: int = 10) -> dict[str, list]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    counts = [0 for _ in range(bins)]
    for value in confidences:
        idx = min(bins - 1, max(0, int(value * bins)))
        counts[idx] += 1
    edges = [i / bins for i in range(bins + 1)]
    return {"bins": edges, "counts": counts}


def entropy_metrics(probabilities: list[dict[str, float]]) -> dict[str, float]:
    entropies: list[float] = []
    for row in probabilities:
        entropy = 0.0
        for value in row.values():
            if value > 0:
                entropy -= value * log(value)
        entropies.append(entropy)
    if not entropies:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
    entropies_sorted = sorted(entropies)
    p50 = entropies_sorted[int(0.5 * (len(entropies_sorted) - 1))]
    p90 = entropies_sorted[int(0.9 * (len(entropies_sorted) - 1))]
    return {
        "mean": sum(entropies) / len(entropies),
        "p50": p50,
        "p90": p90,
    }


def stage_distribution(stages: list[str]) -> dict[str, float]:
    counts = Counter(stages)
    total = sum(counts.values())
    if total == 0:
        return {stage: 0.0 for stage in counts}
    return {stage: count / total for stage, count in counts.items()}


def average_confidence(confidences: list[float]) -> float:
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def per_class_frequency(stages: list[str]) -> list[dict[str, object]]:
    counts = Counter(stages)
    total = sum(counts.values())
    rows: list[dict[str, object]] = []
    for label in sorted(counts):
        count = counts[label]
        frequency = count / total if total > 0 else 0.0
        rows.append({"label": label, "count": count, "frequency": frequency})
    return rows


def prediction_distribution(stages: list[str]) -> dict[str, float]:
    return stage_distribution(stages)


def night_summary_metrics(
    stages: list[str], epoch_minutes: float = 0.5
) -> dict[str, object]:
    total_minutes = len(stages) * epoch_minutes
    sleep_minutes = sum(1 for stage in stages if stage != "W") * epoch_minutes
    sleep_efficiency = (sleep_minutes / total_minutes) if total_minutes > 0 else 0.0

    sleep_latency = None
    waso = None
    if stages:
        try:
            onset_index = next(i for i, stage in enumerate(stages) if stage != "W")
            sleep_latency = onset_index * epoch_minutes
            waso = (
                sum(1 for stage in stages[onset_index:] if stage == "W") * epoch_minutes
            )
        except StopIteration:
            sleep_latency = None
            waso = None

    stage_counts = Counter(stages)
    stage_proportions = {
        stage: (count * epoch_minutes) / total_minutes if total_minutes > 0 else 0.0
        for stage, count in stage_counts.items()
    }

    return {
        "total_minutes": total_minutes,
        "total_sleep_minutes": sleep_minutes,
        "sleep_efficiency": sleep_efficiency,
        "sleep_latency_minutes": sleep_latency,
        "waso_minutes": waso,
        "stage_proportions": stage_proportions,
    }


def night_summary_delta(
    predicted: dict[str, object],
    ground_truth: dict[str, object],
) -> dict[str, object]:
    delta: dict[str, object] = {}
    for key in [
        "total_minutes",
        "total_sleep_minutes",
        "sleep_efficiency",
        "sleep_latency_minutes",
        "waso_minutes",
    ]:
        pred_val = predicted.get(key)
        truth_val = ground_truth.get(key)
        if not isinstance(pred_val, (int, float)) or not isinstance(
            truth_val, (int, float)
        ):
            delta[key] = None
        else:
            delta[key] = float(pred_val) - float(truth_val)

    delta_stage = {}
    pred_stage = predicted.get("stage_proportions")
    truth_stage = ground_truth.get("stage_proportions")
    if not isinstance(pred_stage, dict):
        pred_stage = {}
    if not isinstance(truth_stage, dict):
        truth_stage = {}
    for stage in set(pred_stage) | set(truth_stage):
        delta_stage[stage] = float(pred_stage.get(stage, 0.0) or 0.0) - float(
            truth_stage.get(stage, 0.0) or 0.0
        )
    delta["stage_proportions"] = delta_stage
    return delta
