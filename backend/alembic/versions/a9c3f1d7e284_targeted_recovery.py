"""targeted object recovery tables

Revision ID: a9c3f1d7e284
Revises: f7a2c6e91b04
Create Date: 2026-11-01 20:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a9c3f1d7e284"
down_revision: Union[str, None] = "f7a2c6e91b04"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    if "recovery_experiments" not in existing:
        op.create_table(
            "recovery_experiments",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("run_id", sa.String(36), sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("source_revision_id", sa.String(36), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("mode", sa.String(16), nullable=False),
            sa.Column("cancel_requested", sa.Boolean(), nullable=False),
            sa.Column("generation", sa.Integer(), nullable=False),
            sa.Column("targets_json", sa.JSON(), nullable=False),
            sa.Column("limits_json", sa.JSON(), nullable=False),
            sa.Column("plan_json", sa.JSON(), nullable=False),
            sa.Column("summary_json", sa.JSON(), nullable=False),
            sa.Column("applied_revision_id", sa.String(36), nullable=True),
            sa.Column("error", sa.Text(), nullable=False),
            sa.Column("recorded_cost_usd", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_recovery_experiments_run_id", "recovery_experiments", ["run_id"])
        op.create_index("ix_recovery_experiments_status", "recovery_experiments", ["status"])
    if "recovery_requests" not in existing:
        op.create_table(
            "recovery_requests",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "experiment_id", sa.String(36), sa.ForeignKey("recovery_experiments.id", ondelete="CASCADE"), nullable=False
            ),
            sa.Column("run_id", sa.String(36), nullable=False),
            sa.Column("cache_key", sa.String(64), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("submission", sa.String(32), nullable=False),
            sa.Column("source", sa.String(16), nullable=False),
            sa.Column("media_json", sa.JSON(), nullable=False),
            sa.Column("model_request_id", sa.String(36), nullable=False),
            sa.Column("provider_request_id", sa.String(200), nullable=False),
            sa.Column("latency_ms", sa.Integer(), nullable=False),
            sa.Column("input_tokens", sa.Integer(), nullable=False),
            sa.Column("output_tokens", sa.Integer(), nullable=False),
            sa.Column("billed_frames", sa.Integer(), nullable=False),
            sa.Column("cost_usd", sa.Float(), nullable=True),
            sa.Column("error", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("experiment_id", "cache_key", name="uq_recovery_request_key"),
        )
        op.create_index("ix_recovery_requests_experiment_id", "recovery_requests", ["experiment_id"])
        op.create_index("ix_recovery_requests_run_id", "recovery_requests", ["run_id"])
        op.create_index("ix_recovery_requests_cache_key", "recovery_requests", ["cache_key"])
    if "recovery_proposals" not in existing:
        op.create_table(
            "recovery_proposals",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "experiment_id", sa.String(36), sa.ForeignKey("recovery_experiments.id", ondelete="CASCADE"), nullable=False
            ),
            sa.Column("run_id", sa.String(36), nullable=False),
            sa.Column("generation", sa.Integer(), nullable=False),
            sa.Column("target_id", sa.String(32), nullable=False),
            sa.Column("object_local_id", sa.String(80), nullable=False),
            sa.Column("time_start", sa.Float(), nullable=False),
            sa.Column("time_end", sa.Float(), nullable=False),
            sa.Column("decision", sa.String(16), nullable=False),
            sa.Column("rows_json", sa.JSON(), nullable=False),
            sa.Column("metrics_json", sa.JSON(), nullable=False),
            sa.Column("applied_revision_id", sa.String(36), nullable=True),
            sa.Column("decided_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_recovery_proposals_experiment_id", "recovery_proposals", ["experiment_id"])
        op.create_index("ix_recovery_proposals_run_id", "recovery_proposals", ["run_id"])


def downgrade() -> None:
    op.drop_table("recovery_proposals")
    op.drop_table("recovery_requests")
    op.drop_table("recovery_experiments")
