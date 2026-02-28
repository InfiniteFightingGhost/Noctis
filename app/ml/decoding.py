from __future__ import annotations

import numpy as np


FOUR_CLASS_ORDER = ["W", "Light", "Deep", "REM"]


def transition_penalty_matrix(labels: list[str]) -> np.ndarray:
    n = len(labels)
    penalties = np.full((n, n), 1.4, dtype=np.float32)
    for i in range(n):
        penalties[i, i] = 0.0
    index = {label: idx for idx, label in enumerate(labels)}

    def low(a: str, b: str) -> None:
        if a in index and b in index:
            penalties[index[a], index[b]] = 0.2
            penalties[index[b], index[a]] = 0.2

    def high(a: str, b: str) -> None:
        if a in index and b in index:
            penalties[index[a], index[b]] = 4.0

    low("W", "Light")
    low("Light", "Deep")
    low("Light", "REM")
    high("W", "Deep")
    high("Deep", "REM")
    high("REM", "Deep")
    return penalties


def viterbi_decode_probabilities(probabilities: np.ndarray, labels: list[str]) -> np.ndarray:
    return viterbi_decode_probabilities_with_penalties(
        probabilities, labels, transition_penalties=None
    )


def viterbi_decode_probabilities_with_penalties(
    probabilities: np.ndarray,
    labels: list[str],
    transition_penalties: np.ndarray | None,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probabilities must be 2D")
    if probs.shape[0] == 0:
        return np.asarray([], dtype=np.int64)
    if transition_penalties is None:
        penalties = transition_penalty_matrix(labels)
    else:
        penalties = np.asarray(transition_penalties, dtype=np.float64)
        if penalties.shape != (probs.shape[1], probs.shape[1]):
            raise ValueError("transition_penalties must match class dimensions")
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    t_steps, n = log_probs.shape
    scores = np.full((t_steps, n), -np.inf, dtype=np.float64)
    back = np.zeros((t_steps, n), dtype=np.int64)
    scores[0] = log_probs[0]

    for t in range(1, t_steps):
        for curr in range(n):
            transition_scores = scores[t - 1] - penalties[:, curr]
            best_prev = int(np.argmax(transition_scores))
            back[t, curr] = best_prev
            scores[t, curr] = transition_scores[best_prev] + log_probs[t, curr]

    path = np.zeros(t_steps, dtype=np.int64)
    path[-1] = int(np.argmax(scores[-1]))
    for t in range(t_steps - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path
