from __future__ import annotations

from pathlib import Path
import h5py

from dreem_extractor.channels.resolver import resolve_channels
from dreem_extractor.config import ExtractConfig
from dreem_extractor.io.h5_reader import list_dataset_paths
from dreem_extractor.models import RecordManifest


def build_manifest(path: str | Path, h5file: h5py.File, config: ExtractConfig) -> RecordManifest:
    dataset_paths = list_dataset_paths(h5file)
    hypnogram_path = "/hypnogram" if "/hypnogram" in dataset_paths else None
    channel_map, fs_map = resolve_channels(h5file, config)
    start_time = _read_start_time(h5file)
    return RecordManifest(
        record_id=Path(path).stem,
        hypnogram_path=hypnogram_path,
        start_time=start_time,
        paths_present=dataset_paths,
        channel_map=channel_map,
        fs_map=fs_map,
    )


def _read_start_time(h5file: h5py.File) -> str | None:
    if "start_time" in h5file.attrs:
        value = h5file.attrs["start_time"]
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)
    if "start_time" in h5file:
        value = h5file["start_time"][()]
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)
    return None
