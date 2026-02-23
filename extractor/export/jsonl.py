from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from extractor.extract import EpochRecord


def write_jsonl(records: Iterable[EpochRecord], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "recording_id": record.recording_id,
                "epoch_index": record.epoch_index,
                "stage": record.stage,
                "features": record.features,
            }
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
