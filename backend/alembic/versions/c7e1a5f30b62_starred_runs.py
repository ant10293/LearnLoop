"""starred runs

Revision ID: c7e1a5f30b62
Revises: b4d7e2c15a90, d2f6a8c3e915
Create Date: 2026-11-01 23:30:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c7e1a5f30b62"
down_revision: Union[str, Sequence[str], None] = ("b4d7e2c15a90", "d2f6a8c3e915")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("analysis_runs")}
    if "starred" in columns:
        return
    with op.batch_alter_table("analysis_runs") as batch:
        batch.add_column(sa.Column("starred", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    with op.batch_alter_table("analysis_runs") as batch:
        batch.drop_column("starred")
