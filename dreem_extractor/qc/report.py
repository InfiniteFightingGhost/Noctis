from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_qc_report(qc: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(qc, handle, indent=2)
        handle.write("\n")
