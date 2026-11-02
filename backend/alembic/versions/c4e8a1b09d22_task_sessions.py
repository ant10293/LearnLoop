"""task sessions and cross-clip identity

Revision ID: c4e8a1b09d22
Revises: 4764a7701322
Create Date: 2026-10-21 20:40:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c4e8a1b09d22"
down_revision: Union[str, None] = "4764a7701322"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "projects" in tables and "is_task" not in _columns(inspector, "projects"):
        op.add_column("projects", sa.Column("is_task", sa.Boolean(), nullable=False, server_default=sa.false()))
    if "video_assets" in tables:
        present = _columns(inspector, "video_assets")
        additions = [
            ("session_id", sa.String(length=36), None),
            ("research_mode", sa.String(length=64), ""),
            ("capture_time_raw", sa.String(length=120), ""),
            ("capture_time_utc", sa.String(length=40), ""),
            ("capture_timezone", sa.String(length=40), ""),
            ("timestamp_source", sa.String(length=40), "absent"),
            ("timestamp_reliability", sa.String(length=32), "absent"),
            ("timestamp_note", sa.Text(), ""),
            ("file_mtime", sa.String(length=40), ""),
            ("order_index", sa.Float(), None),
            ("order_locked", sa.Boolean(), False),
            ("order_method", sa.String(length=32), ""),
            ("order_ambiguous", sa.Boolean(), False),
        ]
        for name, column_type, default in additions:
            if name not in present:
                server_default = None if default is None else sa.text(_sql_default(default))
                op.add_column(
                    "video_assets",
                    sa.Column(name, column_type, nullable=True, server_default=server_default),
                )
    if "recording_sessions" not in tables:
        op.create_table(
            "recording_sessions",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("project_id", sa.String(length=36), nullable=False),
            sa.Column("name", sa.String(length=200), nullable=False),
            sa.Column("ordinal", sa.Integer(), nullable=False),
            sa.Column("locked", sa.Boolean(), nullable=False),
            sa.Column("notes", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_recording_sessions_project_id", "recording_sessions", ["project_id"])
    if "canonical_objects" not in tables:
        op.create_table(
            "canonical_objects",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("project_id", sa.String(length=36), nullable=False),
            sa.Column("local_id", sa.String(length=80), nullable=False),
            sa.Column("label", sa.String(length=300), nullable=False),
            sa.Column("description", sa.Text(), nullable=False),
            sa.Column("color", sa.String(length=16), nullable=False),
            sa.Column("uncertain", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_canonical_objects_project_id", "canonical_objects", ["project_id"])
    if "identity_links" not in tables:
        op.create_table(
            "identity_links",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("project_id", sa.String(length=36), nullable=False),
            sa.Column("canonical_id", sa.String(length=36), nullable=False),
            sa.Column("video_id", sa.String(length=36), nullable=False),
            sa.Column("run_id", sa.String(length=36), nullable=False),
            sa.Column("object_local_id", sa.String(length=80), nullable=False),
            sa.Column("label", sa.String(length=300), nullable=False),
            sa.Column("state_before", sa.Text(), nullable=False),
            sa.Column("state_after", sa.Text(), nullable=False),
            sa.Column("image_url", sa.String(length=300), nullable=False),
            sa.Column("method", sa.String(length=40), nullable=False),
            sa.Column("score", sa.Float(), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("uncertain", sa.Boolean(), nullable=False),
            sa.Column("note", sa.Text(), nullable=False),
            sa.Column("evidence_class", sa.String(length=40), nullable=False),
            sa.Column("provenance", sa.String(length=40), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_identity_links_project_id", "identity_links", ["project_id"])
    if "inter_session_transitions" not in tables:
        op.create_table(
            "inter_session_transitions",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("project_id", sa.String(length=36), nullable=False),
            sa.Column("canonical_id", sa.String(length=36), nullable=False),
            sa.Column("from_video_id", sa.String(length=36), nullable=False),
            sa.Column("to_video_id", sa.String(length=36), nullable=False),
            sa.Column("observed_change", sa.Text(), nullable=False),
            sa.Column("hypothesis", sa.Text(), nullable=False),
            sa.Column("from_state", sa.Text(), nullable=False),
            sa.Column("to_state", sa.Text(), nullable=False),
            sa.Column("evidence_class", sa.String(length=40), nullable=False),
            sa.Column("provenance", sa.String(length=40), nullable=False),
            sa.Column("verified", sa.Boolean(), nullable=False),
            sa.Column("gap_note", sa.Text(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_inter_session_transitions_project_id", "inter_session_transitions", ["project_id"])
    if "project_revisions" not in tables:
        op.create_table(
            "project_revisions",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("project_id", sa.String(length=36), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("parent_id", sa.String(length=36), nullable=True),
            sa.Column("trigger", sa.String(length=200), nullable=False),
            sa.Column("change_summary", sa.Text(), nullable=False),
            sa.Column("document_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_project_revisions_project_id", "project_revisions", ["project_id"])
    if "order_hypotheses" not in tables:
        op.create_table(
            "order_hypotheses",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("project_id", sa.String(length=36), nullable=False),
            sa.Column("method", sa.String(length=32), nullable=False),
            sa.Column("ranked_json", sa.JSON(), nullable=False),
            sa.Column("evidence_json", sa.JSON(), nullable=False),
            sa.Column("accepted", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_order_hypotheses_project_id", "order_hypotheses", ["project_id"])
    if "project_ground_truth" not in tables:
        op.create_table(
            "project_ground_truth",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("project_id", sa.String(length=36), nullable=False),
            sa.Column("name", sa.String(length=300), nullable=False),
            sa.Column("document_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_project_ground_truth_project_id", "project_ground_truth", ["project_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    for name in (
        "project_ground_truth",
        "order_hypotheses",
        "project_revisions",
        "inter_session_transitions",
        "identity_links",
        "canonical_objects",
        "recording_sessions",
    ):
        if name in tables:
            op.drop_table(name)


def _columns(inspector, table: str) -> set[str]:
    return {column["name"] for column in inspector.get_columns(table)}


def _sql_default(value) -> str:
    if value is True:
        return "1"
    if value is False:
        return "0"
    return "'" + str(value).replace("'", "''") + "'"
