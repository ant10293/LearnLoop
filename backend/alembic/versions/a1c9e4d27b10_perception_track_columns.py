"""track status on spatial points

Revision ID: a1c9e4d27b10
Revises: e3b91c4a7f20
Create Date: 2026-10-25 18:20:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1c9e4d27b10"
down_revision: Union[str, None] = "e3b91c4a7f20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "spatial_track_points" not in inspector.get_table_names():
        return
    present = {column["name"] for column in inspector.get_columns("spatial_track_points")}
    if "track_status" not in present:
        op.add_column("spatial_track_points", sa.Column("track_status", sa.String(length=32), nullable=True))
    if "association_score" not in present:
        op.add_column("spatial_track_points", sa.Column("association_score", sa.Float(), nullable=True))
    if "detector_id" not in present:
        op.add_column("spatial_track_points", sa.Column("detector_id", sa.String(length=80), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "spatial_track_points" not in inspector.get_table_names():
        return
    present = {column["name"] for column in inspector.get_columns("spatial_track_points")}
    if "detector_id" in present:
        op.drop_column("spatial_track_points", "detector_id")
    if "association_score" in present:
        op.drop_column("spatial_track_points", "association_score")
    if "track_status" in present:
        op.drop_column("spatial_track_points", "track_status")
