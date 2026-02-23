from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


@dataclass
class Signal:
    name: str
    path: str
    data: np.ndarray
    fs: float | None


def open_h5(path: str | Path) -> h5py.File:
    return h5py.File(path, "r")


def load_hypnogram(h5file: h5py.File) -> np.ndarray | None:
    if "hypnogram" in h5file:
        return np.asarray(h5file["hypnogram"])
    for path, dataset in iter_datasets(h5file):
        name = path.lower()
        if "hypnogram" in name or "stages" in name or name.endswith("/stage"):
            return np.asarray(dataset)
    return None


def discover_signals(
    h5file: h5py.File, fs_override: float | None = None
) -> dict[str, dict[str, Any]]:
    signals: dict[str, dict[str, Any]] = {}
    if "signals" in h5file:
        dataset_iter = iter_datasets(h5file["signals"], "/signals")
    else:
        dataset_iter = iter_datasets(h5file)
    for path, dataset in dataset_iter:
        name = path.split("/")[-1]
        kind = detect_kind(name)
        if kind is None:
            continue
        data = np.asarray(dataset)
        if data.ndim > 1:
            data = data[:, 0]
        fs = fs_override or read_sampling_rate(dataset)
        signals[kind] = {
            "name": name,
            "path": path,
            "data": data,
            "fs": fs,
        }
    return signals


def iter_datasets(
    group: h5py.Group, prefix: str = ""
) -> Iterator[tuple[str, h5py.Dataset]]:
    for key, item in group.items():
        path = f"{prefix}/{key}" if prefix else f"/{key}"
        if isinstance(item, h5py.Dataset):
            yield path, item
        elif isinstance(item, h5py.Group):
            yield from iter_datasets(item, path)


def detect_kind(name: str) -> str | None:
    lname = name.lower()
    if "ecg" in lname or "ekg" in lname:
        return "ecg"
    if "emg" in lname:
        return "emg"
    if "resp" in lname or "breath" in lname:
        return "resp"
    if "vib" in lname or "vibration" in lname:
        return "vib"
    if "acc" in lname or "gyro" in lname or "motion" in lname or "move" in lname:
        return "move"
    if "bed" in lname or "in_bed" in lname:
        return "bed"
    if "apnea" in lname:
        return "apnea"
    return None


def read_sampling_rate(dataset: h5py.Dataset) -> float | None:
    for key in ("sampling_rate", "sample_rate", "fs", "sr", "freq"):
        if key in dataset.attrs:
            value = dataset.attrs[key]
            if isinstance(value, (list, tuple, np.ndarray)):
                value = value[0]
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None
