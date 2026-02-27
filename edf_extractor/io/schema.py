from __future__ import annotations

from pathlib import Path

from edf_extractor.channels.resolver import resolve_channels
from edf_extractor.config import ExtractConfig
from edf_extractor.io.edf_reader import EDFSignal
from edf_extractor.models import RecordManifest


def build_manifest(
    edf_path: str | Path,
    signals: list[EDFSignal],
    config: ExtractConfig,
    cap_path: str | Path | None = None,
    start_time: str | None = None,
) -> RecordManifest:
    channel_map, fs_map, index_map = resolve_channels(signals, config)
    cap_ref = str(Path(cap_path)) if cap_path is not None else None
    return RecordManifest(
        record_id=Path(edf_path).stem,
        hypnogram_ref=cap_ref,
        start_time=start_time,
        paths_present=[signal.label for signal in signals],
        channel_map=channel_map,
        fs_map=fs_map,
        channel_index_map=index_map,
    )
