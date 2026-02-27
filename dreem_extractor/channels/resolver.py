from __future__ import annotations

import fnmatch
import h5py

from dreem_extractor.config import ExtractConfig
from dreem_extractor.io.h5_reader import iter_datasets
from extractor_hardened.errors import ExtractionError


def resolve_channels(
    h5file: h5py.File, config: ExtractConfig
) -> tuple[dict[str, str], dict[str, float]]:
    paths = [path for path, _ in iter_datasets(h5file)]
    channel_map: dict[str, str] = {}
    fs_map: dict[str, float] = {}
    for logical, patterns in config.channel_patterns.items():
        match = _match_first(logical, paths, patterns)
        if match is None:
            continue
        channel_map[logical] = match
        dataset = h5file[match]
        fs = _read_fs(dataset, config.fs_attr_keys)
        if fs is None:
            fs = _read_fs_from_group(h5file, match, config.fs_attr_keys)
        if fs is not None:
            fs_map[logical] = fs
    return channel_map, fs_map


def _match_first(logical: str, paths: list[str], patterns: list[str]) -> str | None:
    strict_ambiguity = logical in {"ecg"}
    for pattern in patterns:
        matches = sorted(path for path in paths if fnmatch.fnmatch(path, pattern))
        if len(matches) > 1:
            if strict_ambiguity:
                raise ExtractionError(
                    code="E_CHANNEL_AMBIGUOUS",
                    message="Multiple H5 channels matched same pattern",
                    details={"logical": logical, "pattern": pattern, "matches": matches},
                )
            return matches[0]
        if matches:
            return matches[0]
    return None


def _read_fs(dataset: h5py.Dataset, keys: list[str]) -> float | None:
    for key in keys:
        if key in dataset.attrs:
            value = dataset.attrs[key]
            if isinstance(value, (list, tuple)):
                value = value[0]
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _read_fs_from_group(h5file: h5py.File, dataset_path: str, keys: list[str]) -> float | None:
    parts = dataset_path.strip("/").split("/")
    for idx in range(len(parts) - 1, 0, -1):
        group_path = "/" + "/".join(parts[:idx])
        if group_path in h5file:
            group = h5file[group_path]
            if isinstance(group, h5py.Group):
                for key in keys:
                    if key in group.attrs:
                        value = group.attrs[key]
                        if isinstance(value, (list, tuple)):
                            value = value[0]
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            continue
    return None
