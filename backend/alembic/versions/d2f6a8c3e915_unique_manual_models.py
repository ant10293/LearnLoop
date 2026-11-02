"""one manual model row per provider and model id

Revision ID: d2f6a8c3e915
Revises: b8d4e2a61c30
Create Date: 2026-10-29 02:40:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d2f6a8c3e915"
down_revision: Union[str, None] = "b8d4e2a61c30"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX = "uq_manual_model_provider_model"
DEDUPE_MANUAL_MODELS = """
DELETE FROM manual_models
WHERE id NOT IN (
    SELECT id FROM (
        SELECT id, ROW_NUMBER() OVER (
            PARTITION BY provider_id, model_id ORDER BY created_at, id
        ) AS position
        FROM manual_models
    ) AS ranked WHERE position = 1
)
"""


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "manual_models" not in inspector.get_table_names():
        return
    if any(index["name"] == INDEX for index in inspector.get_indexes("manual_models")):
        return
    op.execute(DEDUPE_MANUAL_MODELS)
    op.create_index(INDEX, "manual_models", ["provider_id", "model_id"], unique=True)


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "manual_models" not in inspector.get_table_names():
        return
    if any(index["name"] == INDEX for index in inspector.get_indexes("manual_models")):
        op.drop_index(INDEX, table_name="manual_models")
