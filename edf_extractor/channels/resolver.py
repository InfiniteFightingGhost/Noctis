from __future__ import annotations

from edf_extractor.config import ExtractConfig
from edf_extractor.io.edf_reader import EDFSignal
from extractor_hardened.errors import ExtractionError


def resolve_channels(
    signals: list[EDFSignal], config: ExtractConfig
) -> tuple[dict[str, str], dict[str, float], dict[str, int]]:
    channel_map: dict[str, str] = {}
    fs_map: dict[str, float] = {}
    index_map: dict[str, int] = {}
    normalized = [_normalize_label(signal.label) for signal in signals]

    for logical, aliases in config.channel_aliases.items():
        idx = _find_first_match(logical, normalized, aliases)
        if idx is None:
            continue
        channel_map[logical] = signals[idx].label
        fs_map[logical] = signals[idx].fs
        index_map[logical] = idx
    return channel_map, fs_map, index_map


def _find_first_match(logical: str, labels: list[str], aliases: list[str]) -> int | None:
    strict_ambiguity = logical in {"ecg"}
    normalized_aliases = [_normalize_label(alias) for alias in aliases]
    for alias in normalized_aliases:
        matched_indices = [idx for idx, label in enumerate(labels) if alias and alias in label]
        if len(matched_indices) > 1:
            if strict_ambiguity:
                matched_labels = {labels[idx] for idx in matched_indices}
                if len(matched_labels) == 1:
                    return matched_indices[0]
                raise ExtractionError(
                    code="E_CHANNEL_AMBIGUOUS",
                    message="Multiple EDF channels matched same alias",
                    details={"logical": logical, "alias": alias, "matches": matched_indices},
                )
            return matched_indices[0]
        if matched_indices:
            return matched_indices[0]
    return None


def _normalize_label(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())
