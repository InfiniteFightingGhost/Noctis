"""create sensor_samples

Revision ID: 88350bd75993
Revises:
Create Date: 2026-02-01 21:28:41.942191

"""

from alembic import op


revision: str = "0001_extensions.py"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    pass


def downgrade() -> None:
    pass
