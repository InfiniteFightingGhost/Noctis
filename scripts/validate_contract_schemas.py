from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    schema_paths = sorted(repo_root.glob("packages/contracts/**/schema.json"))
    if not schema_paths:
        print("No contract schemas found under packages/contracts.")
        return 1

    errors = 0
    for path in schema_paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON: {path} ({exc})")
            errors += 1
        except OSError as exc:
            print(f"Failed to read: {path} ({exc})")
            errors += 1

    if errors:
        print(f"Schema validation failed for {errors} file(s).")
        return 1

    print(f"Validated {len(schema_paths)} contract schema file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
