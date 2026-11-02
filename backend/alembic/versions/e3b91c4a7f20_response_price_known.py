"""record whether a provider response had a catalog price

Revision ID: e3b91c4a7f20
Revises: c4e8a1b09d22
Create Date: 2026-10-22 18:10:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e3b91c4a7f20"
down_revision: Union[str, None] = "c4e8a1b09d22"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "model_responses" not in inspector.get_table_names():
        return
    present = {column["name"] for column in inspector.get_columns("model_responses")}
    if "price_known" in present:
        return
    op.add_column(
        "model_responses",
        sa.Column("price_known", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.execute("UPDATE model_responses SET price_known = 1 WHERE cost_usd > 0")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "model_responses" not in inspector.get_table_names():
        return
    present = {column["name"] for column in inspector.get_columns("model_responses")}
    if "price_known" in present:
        op.drop_column("model_responses", "price_known")
