from __future__ import annotations

import argparse
import os
from pathlib import Path

from extractor.config import ExtractConfig, FEATURE_KEYS
from extractor.export.jsonl import write_jsonl
from extractor.export.metadata import write_feature_order
from extractor.export.npz import build_windows, write_npz
from extractor.extract import extract_recording


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract features from H5 sleep recordings."
    )
    parser.add_argument("--input", required=True, help="Input .h5 file or directory")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--epoch-sec", type=int, default=30)
    parser.add_argument(
        "--drop-unknown-stages", dest="drop_unknown_stages", action="store_true"
    )
    parser.add_argument(
        "--keep-unknown-stages", dest="drop_unknown_stages", action="store_false"
    )
    parser.add_argument("--export-windows", action="store_true", default=False)
    parser.add_argument("--window-len", type=int, default=21)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--label-mode", choices=["center", "sequence"], default="center"
    )
    parser.add_argument("--fs-override", type=float, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.set_defaults(drop_unknown_stages=True)
    return parser.parse_args()


def find_h5_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files: list[Path] = []
    for root, _, filenames in os.walk(input_path):
        for name in filenames:
            if name.lower().endswith(".h5"):
                files.append(Path(root) / name)
    return sorted(files)


def main() -> None:
    args = parse_args()
    config = ExtractConfig(
        epoch_sec=args.epoch_sec,
        drop_unknown_stages=args.drop_unknown_stages,
        export_windows=args.export_windows,
        window_len=args.window_len,
        stride=args.stride,
        label_mode=args.label_mode,
        fs_override=args.fs_override,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = find_h5_files(Path(args.input))
    if not files:
        raise SystemExit("No .h5 files found")

    write_feature_order(FEATURE_KEYS, out_dir)

    for path in files:
        result = extract_recording(path, config)
        records = result.records
        if config.drop_unknown_stages:
            records = [r for r, known in zip(records, result.stage_known) if known]

        jsonl_path = out_dir / f"{result.recording_id}.jsonl"
        write_jsonl(records, jsonl_path)

        if config.export_windows:
            X, y, mask = build_windows(
                result.features,
                result.stages,
                result.stage_known,
                config.window_len,
                config.stride,
                config.label_mode,
                config.drop_unknown_stages,
            )
            npz_path = out_dir / f"{result.recording_id}.npz"
            write_npz(X, y, mask, npz_path)

        if args.verbose:
            print(f"Extracted {len(records)} epochs from {path}")


if __name__ == "__main__":
    main()
