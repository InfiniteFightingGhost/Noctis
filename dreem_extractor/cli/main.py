from __future__ import annotations

import argparse
import os
from pathlib import Path

from dreem_extractor.config import load_config
from dreem_extractor.pipeline import extract_record
from dreem_extractor.serialize.writers import write_manifest, write_record_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dreem_extractor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract", help="Extract features from H5 files"
    )
    extract_parser.add_argument(
        "--input", required=True, help="Input .h5 file or directory"
    )
    extract_parser.add_argument("--output", required=True, help="Output directory")
    extract_parser.add_argument("--config", default=None, help="Path to config YAML")
    extract_parser.add_argument("--manifest", dest="manifest", action="store_true")
    extract_parser.add_argument("--no-manifest", dest="manifest", action="store_false")
    extract_parser.set_defaults(manifest=True)
    extract_parser.add_argument("--verbose", action="store_true", default=False)

    return parser


def find_h5_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files: list[Path] = []
    for root, _, filenames in os.walk(input_path):
        for name in filenames:
            if name.lower().endswith(".h5"):
                files.append(Path(root) / name)
    return sorted(files)


def run_extract(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = find_h5_files(Path(args.input))
    if not files:
        raise SystemExit("No .h5 files found")

    manifest_entries: list[dict[str, object]] = []
    for path in files:
        result = extract_record(path, config)
        outputs = write_record_outputs(result, out_dir)
        manifest_entries.append(
            {
                "record_id": result.record_id,
                "npz": str(outputs["npz"]),
                "metadata": str(outputs["metadata"]),
                "qc": str(outputs["qc"]),
            }
        )
        if args.verbose:
            print(f"Extracted {result.record_id}")

    if args.manifest:
        write_manifest(manifest_entries, out_dir / "manifest.jsonl")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "extract":
        run_extract(args)


if __name__ == "__main__":
    main()
