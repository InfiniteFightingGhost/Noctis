from __future__ import annotations

import argparse
import uuid

from app.db.session import run_with_db_retry
from app.reproducibility.snapshots import verify_snapshot_checksum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a dataset snapshot checksum")
    parser.add_argument("--id", required=True, help="Dataset snapshot id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_id = uuid.UUID(args.id)
    result = run_with_db_retry(
        lambda session: verify_snapshot_checksum(session, snapshot_id=snapshot_id),
        operation_name="verify_snapshot",
    )
    if not result.matches:
        raise SystemExit(
            f"Snapshot checksum mismatch: expected {result.expected_checksum} computed {result.computed_checksum}"
        )
    print(f"snapshot_id={result.snapshot_id} checksum=ok row_count={result.row_count}")


if __name__ == "__main__":
    main()
