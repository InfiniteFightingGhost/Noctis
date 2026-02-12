from __future__ import annotations

import argparse
from pathlib import Path

from app.dataset.builder import build_dataset
from app.dataset.config import load_dataset_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML dataset from TimescaleDB")
    parser.add_argument("--config", required=True, help="Path to dataset config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(Path(args.config))
    build_dataset(config)


if __name__ == "__main__":
    main()
