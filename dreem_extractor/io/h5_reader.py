from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


def open_h5(path: str | Path) -> h5py.File:
    return h5py.File(path, "r")


def iter_datasets(
    group: h5py.Group, prefix: str = ""
) -> Iterator[tuple[str, h5py.Dataset]]:
    for key, item in group.items():
        path = f"{prefix}/{key}" if prefix else f"/{key}"
        if isinstance(item, h5py.Dataset):
            yield path, item
        elif isinstance(item, h5py.Group):
            yield from iter_datasets(item, path)


def list_dataset_paths(h5file: h5py.File) -> list[str]:
    return [path for path, _ in iter_datasets(h5file)]


def read_dataset(h5file: h5py.File, path: str) -> np.ndarray:
    data = np.asarray(h5file[path])
    if data.ndim > 1:
        data = data[:, 0]
    return data


def read_attr(dataset: h5py.Dataset, key: str) -> str | None:
    if key not in dataset.attrs:
        return None
    value = dataset.attrs[key]
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0]
    return str(value)
