from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_feature_order(feature_keys: Iterable[str], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "feature_order.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(feature_keys), handle, indent=2)
        handle.write("\n")
    return path
