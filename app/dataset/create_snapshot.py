from __future__ import annotations

import argparse
from pathlib import Path

from app.dataset.snapshot_config import load_snapshot_config
from app.dataset.snapshots import create_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a dataset snapshot")
    parser.add_argument("--config", required=True, help="Path to snapshot config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_snapshot_config(Path(args.config))
    result = create_snapshot(config)
    print(
        f"snapshot_id={result.snapshot_id} checksum={result.checksum} row_count={result.row_count}"
    )


if __name__ == "__main__":
    main()
