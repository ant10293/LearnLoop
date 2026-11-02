"""dismissed recovery problems

Revision ID: b4d7e2c15a90
Revises: a9c3f1d7e284
Create Date: 2026-11-01 23:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b4d7e2c15a90"
down_revision: Union[str, None] = "a9c3f1d7e284"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if "recovery_dismissals" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "recovery_dismissals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_id", sa.String(32), nullable=False),
        sa.Column("object_local_id", sa.String(80), nullable=False),
        sa.Column("time_start", sa.Float(), nullable=False),
        sa.Column("time_end", sa.Float(), nullable=False),
        sa.Column("reasons_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("run_id", "target_id", name="uq_recovery_dismissal_target"),
    )
    op.create_index("ix_recovery_dismissals_run_id", "recovery_dismissals", ["run_id"])


def downgrade() -> None:
    op.drop_table("recovery_dismissals")
