from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

from tests.integration.utils import migrate_database


@pytest.mark.skipif(
    os.getenv("INTEGRATION_TEST_DATABASE_URL") is None,
    reason="Integration DB not configured",
)
def test_backup_restore_validation() -> None:
    if not shutil.which("pg_dump") or not shutil.which("pg_restore"):
        pytest.skip("pg_dump/pg_restore not available")
    database_url = os.environ["INTEGRATION_TEST_DATABASE_URL"]
    os.environ["DATABASE_URL"] = database_url
    migrate_database(database_url)

    with tempfile.TemporaryDirectory() as tmpdir:
        backup_path = os.path.join(tmpdir, "backup.dump")
        subprocess.run(
            [sys.executable, "-m", "app.ops.backup", "--output", backup_path],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "app.ops.restore_test", backup_path],
            check=True,
        )
