"""let a video exist without a project

Revision ID: f7a2c6e91b04
Revises: e3b91c4a7f20
Create Date: 2026-10-23 00:50:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f7a2c6e91b04"
down_revision: Union[str, None] = "e3b91c4a7f20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for table in ("video_assets", "analysis_runs"):
        if table not in inspector.get_table_names():
            continue
        column = next((item for item in inspector.get_columns(table) if item["name"] == "project_id"), None)
        if column is None or column["nullable"]:
            continue
        with op.batch_alter_table(table) as batch:
            batch.alter_column("project_id", existing_type=sa.String(length=36), nullable=True)


def downgrade() -> None:
    """Unassigned videos have no project to return to."""
    return
