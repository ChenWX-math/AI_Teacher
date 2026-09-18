"""Create sessions, messages and teaching_states."""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260918_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(length=96), primary_key=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("settings", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            "session_id",
            sa.String(length=96),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_messages_session_id", "messages", ["session_id"])
    op.create_table(
        "teaching_states",
        sa.Column(
            "session_id",
            sa.String(length=96),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("subject", sa.String(length=64), nullable=True),
        sa.Column("current_topic", sa.Text(), nullable=True),
        sa.Column("current_exercise", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("reference_answer", sa.Text(), nullable=True),
        sa.Column("last_grade", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("conversation_summary", sa.Text(), nullable=False),
        sa.Column("summarized_message_count", sa.Integer(), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_table("teaching_states")
    op.drop_index("ix_messages_session_id", table_name="messages")
    op.drop_table("messages")
    op.drop_table("sessions")
