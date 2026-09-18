import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import core.chat_service as chat_service_module
from core.chat_service import TeacherChatService
from core.database import Base
from core.repositories.postgres_repository import PostgresStorageRepository
from core.teacher_agent import TeacherAgentResult


class UnusedModel:
    pass


@pytest.fixture
def repository():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    result = PostgresStorageRepository(engine)
    result.create_session(session_id="lesson")
    return result


def build_service(repository, monkeypatch, fake_run):
    monkeypatch.setattr(chat_service_module, "build_teacher_agent", lambda **_kwargs: object())
    monkeypatch.setattr(chat_service_module, "run_teacher_agent", fake_run)
    return TeacherChatService(
        repository,
        knowledge_base_loader=lambda: None,
        llm_factory=lambda **_kwargs: UnusedModel(),
    )


def test_chat_commits_messages_and_state_in_one_transaction(repository, monkeypatch):
    def fake_run(_agent, **kwargs):
        kwargs["teaching_state"].current_topic = "二次函数"
        kwargs["state_saver"](kwargs["teaching_state"])
        kwargs["history"].add_messages(
            [HumanMessage(content=kwargs["user_input"]), AIMessage(content="测试回答")]
        )
        return TeacherAgentResult(answer="测试回答")

    service = build_service(repository, monkeypatch, fake_run)

    result = service.chat(
        session_id="lesson",
        message="讲二次函数",
        subject="数学",
        gender="女",
        personality="耐心",
    )

    assert result.answer == "测试回答"
    assert len(repository.load_messages("lesson")) == 2
    assert repository.load_teaching_state("lesson")["current_topic"] == "二次函数"


def test_chat_rolls_back_messages_and_state_together(repository, monkeypatch):
    def fake_run(_agent, **kwargs):
        kwargs["teaching_state"].current_topic = "不应保存"
        kwargs["state_saver"](kwargs["teaching_state"])
        kwargs["history"].add_messages(
            [HumanMessage(content=kwargs["user_input"]), AIMessage(content="半成品")]
        )
        raise RuntimeError("agent failed")

    service = build_service(repository, monkeypatch, fake_run)

    with pytest.raises(RuntimeError, match="agent failed"):
        service.chat(
            session_id="lesson",
            message="讲二次函数",
            subject="数学",
            gender="女",
            personality="耐心",
        )

    assert repository.load_messages("lesson") == []
    assert repository.load_teaching_state("lesson") is None
