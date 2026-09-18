"""SQLAlchemy 2.x PostgreSQL 存储实现。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, create_engine, delete, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from core.config import get_required
from core.database import MessageModel, SessionModel, TeachingStateModel
from core.repositories.base import SessionRecord, StorageRepository
from core.repositories.json_repository import create_session_id


class PostgresStorageRepository(StorageRepository):
    """生产使用 PostgreSQL；测试可注入 SQLAlchemy Engine。"""

    def __init__(self, engine: Engine, db_session: Session | None = None):
        self.engine = engine
        self._db_session = db_session
        self._session_factory = sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def from_config(cls) -> "PostgresStorageRepository":
        engine = create_engine(get_required("DATABASE_URL"), pool_pre_ping=True)
        return cls(engine)

    @contextmanager
    def _session_scope(self) -> Iterator[Session]:
        if self._db_session is not None:
            yield self._db_session
            return
        with self._session_factory.begin() as db_session:
            yield db_session

    @contextmanager
    def transaction(self) -> Iterator[StorageRepository]:
        if self._db_session is not None:
            yield self
            return
        with self._session_factory.begin() as db_session:
            yield PostgresStorageRepository(self.engine, db_session=db_session)

    def healthcheck(self) -> bool:
        with self._session_scope() as db_session:
            return db_session.scalar(text("SELECT 1")) == 1

    @staticmethod
    def _record(model: SessionModel) -> SessionRecord:
        return SessionRecord(
            id=model.id,
            title=model.title,
            created_at=model.created_at,
            updated_at=model.updated_at,
            settings=dict(model.settings or {}),
        )

    def create_session(
        self, *, session_id: str | None = None, title: str | None = None
    ) -> SessionRecord:
        identifier = session_id or create_session_id()
        with self._session_scope() as db_session:
            model = SessionModel(id=identifier, title=title or identifier, settings={})
            db_session.add(model)
            db_session.flush()
            db_session.refresh(model)
            return self._record(model)

    def list_sessions(self) -> list[SessionRecord]:
        with self._session_scope() as db_session:
            models = db_session.scalars(
                select(SessionModel).order_by(
                    SessionModel.created_at.desc(), SessionModel.id.desc()
                )
            ).all()
            return [self._record(model) for model in models]

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._session_scope() as db_session:
            model = db_session.get(SessionModel, session_id)
            return self._record(model) if model else None

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> SessionRecord | None:
        with self._session_scope() as db_session:
            model = db_session.get(SessionModel, session_id)
            if model is None:
                return None
            if title is not None:
                model.title = title
            if settings is not None:
                model.settings = {**(model.settings or {}), **settings}
            model.updated_at = datetime.now().astimezone()
            db_session.flush()
            return self._record(model)

    def delete_session(self, session_id: str) -> None:
        with self._session_scope() as db_session:
            db_session.execute(delete(MessageModel).where(MessageModel.session_id == session_id))
            db_session.execute(
                delete(TeachingStateModel).where(TeachingStateModel.session_id == session_id)
            )
            db_session.execute(delete(SessionModel).where(SessionModel.id == session_id))

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._session_scope() as db_session:
            models = db_session.scalars(
                select(MessageModel)
                .where(MessageModel.session_id == session_id)
                .order_by(MessageModel.sequence, MessageModel.id)
            ).all()
            return [dict(model.payload) for model in models]

    @staticmethod
    def _message_model(
        session_id: str, sequence: int, payload: dict[str, Any]
    ) -> MessageModel:
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        content = data.get("content", "")
        return MessageModel(
            session_id=session_id,
            sequence=sequence,
            role=str(payload.get("type") or data.get("type") or "unknown"),
            content=content if isinstance(content, str) else str(content),
            payload=payload,
            message_metadata={
                "additional_kwargs": data.get("additional_kwargs", {}),
                "response_metadata": data.get("response_metadata", {}),
            },
        )

    def replace_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        with self._session_scope() as db_session:
            db_session.scalar(
                select(SessionModel).where(SessionModel.id == session_id).with_for_update()
            )
            db_session.execute(delete(MessageModel).where(MessageModel.session_id == session_id))
            db_session.add_all(
                [
                    self._message_model(session_id, index, payload)
                    for index, payload in enumerate(messages)
                ]
            )
            self._touch_session(db_session, session_id)

    def append_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        if not messages:
            return
        with self._session_scope() as db_session:
            db_session.scalar(
                select(SessionModel).where(SessionModel.id == session_id).with_for_update()
            )
            last_sequence = db_session.scalar(
                select(func.max(MessageModel.sequence)).where(
                    MessageModel.session_id == session_id
                )
            )
            start = int(last_sequence) + 1 if last_sequence is not None else 0
            db_session.add_all(
                [
                    self._message_model(session_id, start + index, payload)
                    for index, payload in enumerate(messages)
                ]
            )
            self._touch_session(db_session, session_id)

    @staticmethod
    def _touch_session(db_session: Session, session_id: str) -> None:
        model = db_session.get(SessionModel, session_id)
        if model is not None:
            model.updated_at = datetime.now().astimezone()

    def load_teaching_state(self, session_id: str) -> dict[str, Any] | None:
        with self._session_scope() as db_session:
            model = db_session.get(TeachingStateModel, session_id)
            if model is None:
                return None
            exercise = dict(model.current_exercise) if model.current_exercise else None
            if exercise is not None:
                exercise["reference_answer"] = model.reference_answer or ""
            return {
                "schema_version": model.schema_version,
                "subject": model.subject,
                "current_topic": model.current_topic,
                "current_exercise": exercise,
                "last_grade": model.last_grade,
                "conversation_summary": model.conversation_summary,
                "summarized_message_count": model.summarized_message_count,
            }

    def save_teaching_state(self, session_id: str, state: dict[str, Any]) -> None:
        exercise = dict(state.get("current_exercise") or {}) or None
        reference_answer = exercise.pop("reference_answer", None) if exercise else None
        with self._session_scope() as db_session:
            model = db_session.get(TeachingStateModel, session_id)
            values = {
                "schema_version": int(state.get("schema_version", 1)),
                "subject": state.get("subject"),
                "current_topic": state.get("current_topic"),
                "current_exercise": exercise,
                "reference_answer": reference_answer,
                "last_grade": state.get("last_grade"),
                "conversation_summary": str(state.get("conversation_summary", "")),
                "summarized_message_count": int(state.get("summarized_message_count", 0)),
            }
            if model is None:
                model = TeachingStateModel(session_id=session_id, **values)
                db_session.add(model)
            else:
                for key, value in values.items():
                    setattr(model, key, value)
                model.updated_at = datetime.now().astimezone()
            self._touch_session(db_session, session_id)

    def clear_teaching_state(self, session_id: str) -> None:
        with self._session_scope() as db_session:
            db_session.execute(
                delete(TeachingStateModel).where(TeachingStateModel.session_id == session_id)
            )
            self._touch_session(db_session, session_id)
