#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_index(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=int)
    return np.array(value, dtype=int)


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_window_set(
    recording_ids: np.ndarray,
    window_end_ts: np.ndarray,
    indices: np.ndarray,
) -> set[tuple[Any, Any]]:
    if indices.size == 0:
        return set()
    rec_slice = recording_ids[indices]
    ts_slice = window_end_ts[indices]
    return {
        (_coerce_scalar(rec_id), _coerce_scalar(ts))
        for rec_id, ts in zip(rec_slice, ts_slice, strict=False)
    }


def _summarize_ids(label: str, recording_ids: set[Any]) -> None:
    ordered = sorted((_coerce_scalar(value) for value in recording_ids), key=str)
    preview = ordered[:10]
    remainder = max(len(ordered) - len(preview), 0)
    print(f"{label} recording_ids ({len(ordered)}): {preview}")
    if remainder:
        print(f"{label} recording_ids remaining: {remainder}")


def _print_split_stats(
    name: str,
    indices: np.ndarray,
    recording_ids: np.ndarray,
    window_end_ts: np.ndarray,
) -> tuple[set[Any], set[tuple[Any, Any]]]:
    windows_count = int(indices.size)
    split_recording_ids = (
        set() if indices.size == 0 else {_coerce_scalar(value) for value in recording_ids[indices]}
    )
    window_pairs = _build_window_set(recording_ids, window_end_ts, indices)
    print(f"{name} windows: {windows_count}")
    print(f"{name} unique recordings: {len(split_recording_ids)}")
    _summarize_ids(name, split_recording_ids)
    return split_recording_ids, window_pairs


def _diff_values(expected: Any, actual: Any, path: str = "") -> list[str]:
    diffs: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        for key in sorted(expected_keys - actual_keys):
            diffs.append(f"missing key in actual: {path}{key}")
        for key in sorted(actual_keys - expected_keys):
            diffs.append(f"unexpected key in actual: {path}{key}")
        for key in sorted(expected_keys & actual_keys):
            diffs.extend(_diff_values(expected[key], actual[key], f"{path}{key}."))
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(f"list length mismatch at {path[:-1]}: {len(expected)} vs {len(actual)}")
            return diffs
        for index, (exp_item, act_item) in enumerate(zip(expected, actual, strict=False)):
            diffs.extend(_diff_values(exp_item, act_item, f"{path}{index}."))
        return diffs
    if expected != actual:
        diffs.append(f"value mismatch at {path[:-1]}: {expected!r} vs {actual!r}")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify dataset split integrity.")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory.")
    parser.add_argument(
        "--expected-split-policy",
        help="Path to JSON file with expected split policy.",
    )
    parser.add_argument(
        "--min-recordings",
        type=int,
        default=3,
        help="Minimum number of recordings per split.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    npz_path = dataset_dir / "dataset.npz"
    metadata_path = dataset_dir / "metadata.json"

    if not npz_path.exists():
        print(f"Missing dataset file: {npz_path}")
        return 1
    if not metadata_path.exists():
        print(f"Missing metadata file: {metadata_path}")
        return 1

    with np.load(npz_path, allow_pickle=True) as data:
        recording_ids = data.get("recording_ids")
        window_end_ts = data.get("window_end_ts")
        if recording_ids is None or window_end_ts is None:
            print("dataset.npz missing recording_ids or window_end_ts")
            return 1

        split_train = _normalize_index(data.get("split_train"))
        split_val = _normalize_index(data.get("split_val"))
        split_test = _normalize_index(data.get("split_test"))

    split_train_ids, split_train_pairs = _print_split_stats(
        "train",
        split_train,
        recording_ids,
        window_end_ts,
    )
    split_val_ids, split_val_pairs = _print_split_stats(
        "val",
        split_val,
        recording_ids,
        window_end_ts,
    )
    split_test_ids, split_test_pairs = _print_split_stats(
        "test",
        split_test,
        recording_ids,
        window_end_ts,
    )

    has_failure = False

    if len(split_train_ids) < args.min_recordings:
        print(f"train split has fewer than {args.min_recordings} recordings")
        has_failure = True
    if len(split_val_ids) < args.min_recordings:
        print(f"val split has fewer than {args.min_recordings} recordings")
        has_failure = True
    if len(split_test_ids) < args.min_recordings:
        print(f"test split has fewer than {args.min_recordings} recordings")
        has_failure = True

    def _report_intersection(label: str, left: set[Any], right: set[Any]) -> None:
        nonlocal has_failure
        intersection = left & right
        print(f"{label} intersection size: {len(intersection)}")
        if intersection:
            has_failure = True

    _report_intersection("recording_ids train/val", split_train_ids, split_val_ids)
    _report_intersection("recording_ids train/test", split_train_ids, split_test_ids)
    _report_intersection("recording_ids val/test", split_val_ids, split_test_ids)

    _report_intersection("windows train/val", split_train_pairs, split_val_pairs)
    _report_intersection("windows train/test", split_train_pairs, split_test_pairs)
    _report_intersection("windows val/test", split_val_pairs, split_test_pairs)

    metadata = _load_json(metadata_path)
    if "split_policy" not in metadata:
        print("metadata.json missing split_policy")
        return 1

    if args.expected_split_policy:
        expected_policy = _load_json(Path(args.expected_split_policy))
        actual_policy = metadata.get("split_policy")
        if actual_policy != expected_policy:
            print("split_policy mismatch with expected policy")
            diffs = _diff_values(expected_policy, actual_policy)
            if diffs:
                print("diff summary:")
                for diff in diffs:
                    print(f"- {diff}")
            return 1

    if has_failure:
        return 1

    print("Split integrity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
