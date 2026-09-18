"""Prevent duplicate message order within a session."""

from collections.abc import Sequence

from alembic import op

revision: str = "20260918_0002"
down_revision: str | None = "20260918_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_message_sequence", "messages", ["session_id", "sequence"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_message_sequence", "messages", type_="unique")
