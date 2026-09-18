"""SQLAlchemy 2.x 数据库基础设施。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JSON_DOCUMENT = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class SessionModel(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    settings: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    messages: Mapped[list["MessageModel"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    teaching_state: Mapped["TeachingStateModel | None"] = relationship(
        back_populates="session", cascade="all, delete-orphan", uselist=False
    )


class MessageModel(Base):
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("session_id", "sequence", name="uq_message_sequence"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT)
    message_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON_DOCUMENT, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    session: Mapped[SessionModel] = relationship(back_populates="messages")


class TeachingStateModel(Base):
    __tablename__ = "teaching_states"

    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True
    )
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    subject: Mapped[str | None] = mapped_column(String(64), nullable=True)
    current_topic: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_exercise: Mapped[dict[str, Any] | None] = mapped_column(
        JSON_DOCUMENT, nullable=True
    )
    reference_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_grade: Mapped[dict[str, Any] | None] = mapped_column(JSON_DOCUMENT, nullable=True)
    conversation_summary: Mapped[str] = mapped_column(Text, default="")
    summarized_message_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    session: Mapped[SessionModel] = relationship(back_populates="teaching_state")
