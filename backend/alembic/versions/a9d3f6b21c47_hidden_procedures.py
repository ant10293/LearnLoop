"""hidden procedures

Revision ID: a9d3f6b21c47
Revises: c7e1a5f30b62
Create Date: 2026-11-04 11:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a9d3f6b21c47"
down_revision: Union[str, Sequence[str], None] = "c7e1a5f30b62"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("projects")}
    if "hidden" in columns:
        return
    with op.batch_alter_table("projects") as batch:
        batch.add_column(sa.Column("hidden", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    with op.batch_alter_table("projects") as batch:
        batch.drop_column("hidden")
