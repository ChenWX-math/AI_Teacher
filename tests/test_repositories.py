"""JSON 与 SQLAlchemy/PostgreSQL Repository 的共享行为契约。"""

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from core.database import Base, TeachingStateModel
from core.repositories.json_repository import JsonStorageRepository
from core.repositories.postgres_repository import PostgresStorageRepository


@pytest.fixture(params=["json", "postgres"])
def repository(request, tmp_path):
    if request.param == "json":
        return JsonStorageRepository(tmp_path)
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return PostgresStorageRepository(engine)


def sample_messages():
    return [
        {"type": "human", "data": {"content": "你好", "response_metadata": {}}},
        {"type": "ai", "data": {"content": "你好，同学", "response_metadata": {}}},
    ]


def sample_state():
    return {
        "schema_version": 1,
        "subject": "数学",
        "current_topic": "二次函数",
        "current_exercise": {
            "topic": "二次函数",
            "difficulty": "基础",
            "question_type": "解答题",
            "question": "求顶点。",
            "reference_answer": "隐藏答案",
            "explanation": "配方法。",
            "created_at": "2026-09-18T00:00:00+08:00",
        },
        "last_grade": None,
        "conversation_summary": "较早对话摘要",
        "summarized_message_count": 4,
    }


def test_repository_session_message_and_state_contract(repository):
    created = repository.create_session(session_id="lesson-1", title="函数课")
    repository.update_session("lesson-1", settings={"subject": "数学"})
    repository.append_messages("lesson-1", sample_messages())
    repository.save_teaching_state("lesson-1", sample_state())

    assert created.id == "lesson-1"
    assert repository.list_sessions()[0].id == "lesson-1"
    assert repository.get_session("lesson-1").settings["subject"] == "数学"
    assert repository.load_messages("lesson-1") == sample_messages()
    assert repository.load_teaching_state("lesson-1") == sample_state()

    repository.clear_messages("lesson-1")
    assert repository.load_messages("lesson-1") == []
    assert repository.load_teaching_state("lesson-1")["current_topic"] == "二次函数"

    repository.delete_session("lesson-1")
    assert repository.get_session("lesson-1") is None
    assert repository.load_messages("lesson-1") == []
    assert repository.load_teaching_state("lesson-1") is None


def test_postgres_repository_keeps_reference_answer_out_of_exercise_json():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    repository = PostgresStorageRepository(engine)
    repository.create_session(session_id="lesson")
    repository.save_teaching_state("lesson", sample_state())

    with Session(engine) as db_session:
        row = db_session.scalar(select(TeachingStateModel))
        assert row.reference_answer == "隐藏答案"
        assert "reference_answer" not in row.current_exercise


def test_postgres_transaction_rolls_back_message_and_state_together():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    repository = PostgresStorageRepository(engine)
    repository.create_session(session_id="lesson")

    with pytest.raises(RuntimeError, match="rollback"):
        with repository.transaction() as transaction:
            transaction.append_messages("lesson", sample_messages())
            transaction.save_teaching_state("lesson", sample_state())
            raise RuntimeError("rollback")

    assert repository.load_messages("lesson") == []
    assert repository.load_teaching_state("lesson") is None
