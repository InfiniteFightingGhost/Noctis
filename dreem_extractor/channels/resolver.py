from __future__ import annotations

import fnmatch
import h5py

from dreem_extractor.config import ExtractConfig
from dreem_extractor.io.h5_reader import iter_datasets


def resolve_channels(
    h5file: h5py.File, config: ExtractConfig
) -> tuple[dict[str, str], dict[str, float]]:
    paths = [path for path, _ in iter_datasets(h5file)]
    channel_map: dict[str, str] = {}
    fs_map: dict[str, float] = {}
    for logical, patterns in config.channel_patterns.items():
        match = _match_first(paths, patterns)
        if match is None:
            continue
        channel_map[logical] = match
        dataset = h5file[match]
        fs = _read_fs(dataset, config.fs_attr_keys)
        if fs is not None:
            fs_map[logical] = fs
    return channel_map, fs_map


def _match_first(paths: list[str], patterns: list[str]) -> str | None:
    for pattern in patterns:
        for path in paths:
            if fnmatch.fnmatch(path, pattern):
                return path
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
