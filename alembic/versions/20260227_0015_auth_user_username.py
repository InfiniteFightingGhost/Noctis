from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260227_0015"
down_revision = "20260226_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("auth_users", sa.Column("username", sa.String(length=64), nullable=True))

    op.execute(
        """
        UPDATE auth_users
        SET username = CONCAT(
            lower(regexp_replace(split_part(email, '@', 1), '[^a-zA-Z0-9_]+', '_', 'g')),
            '_',
            substr(replace(id::text, '-', ''), 1, 6)
        )
        WHERE username IS NULL
        """
    )

    op.alter_column("auth_users", "username", nullable=False)
    op.create_unique_constraint("uq_auth_users_username", "auth_users", ["username"])


def downgrade() -> None:
    op.drop_constraint("uq_auth_users_username", "auth_users", type_="unique")
    op.drop_column("auth_users", "username")
