"""需要显式 TEST_DATABASE_URL 的真实 PostgreSQL Repository 冒烟测试。"""

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from core.repositories.postgres_repository import PostgresStorageRepository

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="未配置 TEST_DATABASE_URL")
def test_real_postgres_repository_round_trip():
    repository = PostgresStorageRepository(create_engine(os.environ["TEST_DATABASE_URL"]))
    session_id = f"integration-{uuid4().hex}"
    try:
        repository.create_session(session_id=session_id, title="integration")
        repository.append_messages(
            session_id,
            [{"type": "human", "data": {"content": "hello"}}],
        )
        repository.save_teaching_state(
            session_id,
            {
                "schema_version": 1,
                "subject": "数学",
                "current_topic": "函数",
                "current_exercise": None,
                "last_grade": None,
                "conversation_summary": "",
                "summarized_message_count": 0,
            },
        )

        assert repository.load_messages(session_id)[0]["data"]["content"] == "hello"
        assert repository.load_teaching_state(session_id)["current_topic"] == "函数"
    finally:
        repository.delete_session(session_id)
