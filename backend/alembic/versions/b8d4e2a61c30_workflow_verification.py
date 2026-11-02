"""learned workflow verification tables

Revision ID: b8d4e2a61c30
Revises: a1c9e4d27b10, f7a2c6e91b04
Create Date: 2026-10-26 01:10:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b8d4e2a61c30"
down_revision: Union[str, Sequence[str], None] = ("a1c9e4d27b10", "f7a2c6e91b04")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "learned_workflow_templates" not in tables:
        op.create_table(
            "learned_workflow_templates",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("project_id", sa.String(length=36), nullable=False),
            sa.Column("family_id", sa.String(length=36), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("parent_id", sa.String(length=36), nullable=True),
            sa.Column("source_run_id", sa.String(length=36), nullable=False),
            sa.Column("source_revision_id", sa.String(length=36), nullable=False),
            sa.Column("source_video_id", sa.String(length=36), nullable=False),
            sa.Column("name", sa.String(length=300), nullable=False),
            sa.Column("acceptance_status", sa.String(length=32), nullable=False),
            sa.Column("document_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_learned_workflow_templates_project_id", "learned_workflow_templates", ["project_id"])
        op.create_index("ix_learned_workflow_templates_family_id", "learned_workflow_templates", ["family_id"])
    if "verification_runs" not in tables:
        op.create_table(
            "verification_runs",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("project_id", sa.String(length=36), nullable=False),
            sa.Column("template_id", sa.String(length=36), nullable=False),
            sa.Column("template_version", sa.Integer(), nullable=False),
            sa.Column("execution_video_id", sa.String(length=36), nullable=False),
            sa.Column("execution_run_id", sa.String(length=36), nullable=False),
            sa.Column("execution_revision_id", sa.String(length=36), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("overall_result", sa.String(length=32), nullable=False),
            sa.Column("config_json", sa.JSON(), nullable=False),
            sa.Column("report_json", sa.JSON(), nullable=False),
            sa.Column("machine_report_json", sa.JSON(), nullable=False),
            sa.Column("error", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("finished_at", sa.DateTime(), nullable=True),
        )
        op.create_index("ix_verification_runs_project_id", "verification_runs", ["project_id"])
        op.create_index("ix_verification_runs_template_id", "verification_runs", ["template_id"])
        op.create_index("ix_verification_runs_execution_run_id", "verification_runs", ["execution_run_id"])
        op.create_index("ix_verification_runs_status", "verification_runs", ["status"])
    if "step_verifications" not in tables:
        op.create_table(
            "step_verifications",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("verification_id", sa.String(length=36), nullable=False),
            sa.Column("step_id", sa.String(length=80), nullable=False),
            sa.Column("execution_state", sa.String(length=32), nullable=False),
            sa.Column("verdict", sa.String(length=32), nullable=False),
            sa.Column("error_category", sa.String(length=64), nullable=False),
            sa.Column("matched_action_ids", sa.JSON(), nullable=False),
            sa.Column("matched_object_ids", sa.JSON(), nullable=False),
            sa.Column("explanation", sa.Text(), nullable=False),
            sa.Column("uncertainty", sa.Text(), nullable=False),
            sa.Column("evidence_json", sa.JSON(), nullable=False),
            sa.Column("history_json", sa.JSON(), nullable=False),
            sa.Column("review_required", sa.Boolean(), nullable=False),
        )
        op.create_index("ix_step_verifications_verification_id", "step_verifications", ["verification_id"])
    if "verification_evidence" not in tables:
        op.create_table(
            "verification_evidence",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("verification_id", sa.String(length=36), nullable=False),
            sa.Column("step_id", sa.String(length=80), nullable=False),
            sa.Column("video_id", sa.String(length=36), nullable=False),
            sa.Column("timestamp", sa.Float(), nullable=True),
            sa.Column("frame_id", sa.String(length=36), nullable=True),
            sa.Column("object_local_ids", sa.JSON(), nullable=False),
            sa.Column("evidence_class", sa.String(length=40), nullable=False),
            sa.Column("provenance", sa.String(length=40), nullable=False),
            sa.Column("explanation", sa.Text(), nullable=False),
            sa.Column("side", sa.String(length=32), nullable=False),
            sa.Column("bbox_json", sa.JSON(), nullable=True),
        )
        op.create_index("ix_verification_evidence_verification_id", "verification_evidence", ["verification_id"])
    if "verification_role_mappings" not in tables:
        op.create_table(
            "verification_role_mappings",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("verification_id", sa.String(length=36), nullable=False),
            sa.Column("role_id", sa.String(length=80), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("object_local_ids", sa.JSON(), nullable=False),
            sa.Column("score", sa.Float(), nullable=False),
            sa.Column("hypotheses_json", sa.JSON(), nullable=False),
            sa.Column("note", sa.Text(), nullable=False),
        )
        op.create_index(
            "ix_verification_role_mappings_verification_id",
            "verification_role_mappings",
            ["verification_id"],
        )
    if "verification_reinspections" not in tables:
        op.create_table(
            "verification_reinspections",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("verification_id", sa.String(length=36), nullable=False),
            sa.Column("step_id", sa.String(length=80), nullable=False),
            sa.Column("question", sa.Text(), nullable=False),
            sa.Column("frame_timestamps", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("output_json", sa.JSON(), nullable=False),
            sa.Column("previous_verdict", sa.String(length=32), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index(
            "ix_verification_reinspections_verification_id",
            "verification_reinspections",
            ["verification_id"],
        )
    if "verification_ground_truth" not in tables:
        op.create_table(
            "verification_ground_truth",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("project_id", sa.String(length=36), nullable=False),
            sa.Column("verification_id", sa.String(length=36), nullable=False),
            sa.Column("name", sa.String(length=300), nullable=False),
            sa.Column("document_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_verification_ground_truth_project_id", "verification_ground_truth", ["project_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    for name in (
        "verification_ground_truth",
        "verification_reinspections",
        "verification_role_mappings",
        "verification_evidence",
        "step_verifications",
        "verification_runs",
        "learned_workflow_templates",
    ):
        if name in tables:
            op.drop_table(name)
