from __future__ import annotations

import argparse
import os
from pathlib import Path

from edf_extractor.config import load_config
from edf_extractor.pipeline import extract_record
from edf_extractor.serialize.writers import write_manifest, write_record_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edf_extractor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract features from EDF/REC files")
    extract_parser.add_argument("--input", required=True, help="Input .edf/.rec file or directory")
    extract_parser.add_argument("--output", required=True, help="Output directory")
    extract_parser.add_argument("--config", default=None, help="Path to config YAML")
    extract_parser.add_argument("--cap", default=None, help="CAP hypnogram file for single EDF")
    extract_parser.add_argument(
        "--cap-dir",
        default=None,
        help="Directory containing CAP files matched by EDF stem",
    )
    extract_parser.add_argument(
        "--isruc-scorer",
        type=int,
        choices=(1, 2),
        default=1,
        help="Preferred ISRUC scorer file suffix (_1 or _2)",
    )
    extract_parser.add_argument("--manifest", dest="manifest", action="store_true")
    extract_parser.add_argument("--no-manifest", dest="manifest", action="store_false")
    extract_parser.set_defaults(manifest=True)
    extract_parser.add_argument("--verbose", action="store_true", default=False)
    return parser


def find_polysom_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files: list[Path] = []
    for root, _, filenames in os.walk(input_path):
        for name in filenames:
            lower = name.lower()
            if lower.endswith(".edf") or lower.endswith(".rec"):
                files.append(Path(root) / name)
    return sorted(files)


def run_extract(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_polysom_files(Path(args.input))
    if not files:
        raise SystemExit("No .edf/.rec files found")

    cap_override = Path(args.cap) if args.cap else None
    cap_dir = Path(args.cap_dir) if args.cap_dir else None

    manifest_entries: list[dict[str, object]] = []
    for edf_path in files:
        cap_path = _resolve_hypnogram_path(
            edf_path,
            cap_override,
            cap_dir,
            preferred_isruc_scorer=args.isruc_scorer,
        )
        if cap_path is None:
            raise SystemExit(f"Missing hypnogram file for {edf_path}")
        result = extract_record(edf_path, cap_path, config)
        outputs = write_record_outputs(result, out_dir)
        manifest_entries.append(
            {
                "record_id": result.record_id,
                "record_dir": str(outputs["record_dir"]),
                "features": str(outputs["features"]),
                "manifest": str(outputs["manifest"]),
            }
        )
        if args.verbose:
            print(f"Extracted {result.record_id}")

    if args.manifest:
        write_manifest(manifest_entries, out_dir / "manifest.jsonl")


def _resolve_hypnogram_path(
    edf_path: Path,
    cap_override: Path | None,
    cap_dir: Path | None,
    preferred_isruc_scorer: int,
) -> Path | None:
    if cap_override is not None:
        return cap_override
    search_roots = [edf_path.parent]
    if cap_dir is not None:
        search_roots.insert(0, cap_dir)

    stem = edf_path.stem
    is_isruc = edf_path.suffix.lower() == ".rec"
    if is_isruc:
        scorer_suffixes = [preferred_isruc_scorer, 2 if preferred_isruc_scorer == 1 else 1]
        for root in search_roots:
            for scorer in scorer_suffixes:
                candidate = root / f"{stem}_{scorer}.txt"
                if candidate.exists():
                    return candidate

    for root in search_roots:
        for suffix in (".txt", ".tsv", ".csv", ".cap"):
            candidate = root / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "extract":
        run_extract(args)


if __name__ == "__main__":
    main()
