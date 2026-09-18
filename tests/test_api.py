from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from api.dependencies import get_chat_service, get_storage_repository
from api.main import app
from core import observability
from core.api_client import ApiStorageRepository
from core.memory import RepositoryChatHistory
from core.repositories.json_repository import JsonStorageRepository
from core.teacher_agent import TeacherAgentResult


class FakeChatService:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return TeacherAgentResult(
            answer="这是测试回答。",
            tool_names=["search_textbook"],
            sources=[{"source": "数学/函数.txt"}],
            duration_ms=12,
        )


def build_client(tmp_path, monkeypatch, fake_service=None):
    repository = JsonStorageRepository(tmp_path / "sessions")
    service = fake_service or FakeChatService()
    monkeypatch.setenv("STORAGE_BACKEND", "json")
    monkeypatch.setenv("APP_LOG_PATH", str(tmp_path / "events.jsonl"))
    observability._event_logger.cache_clear()
    app.dependency_overrides[get_storage_repository] = lambda: repository
    app.dependency_overrides[get_chat_service] = lambda: service
    return TestClient(app), repository, service


def teardown_function():
    app.dependency_overrides.clear()
    observability._event_logger.cache_clear()


def test_health_endpoint(tmp_path, monkeypatch):
    client, _repository, _service = build_client(tmp_path, monkeypatch)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "storage_backend": "json"}
    assert response.headers["X-Request-ID"]


def test_session_endpoints_and_messages(tmp_path, monkeypatch):
    client, repository, _service = build_client(tmp_path, monkeypatch)

    created = client.post(
        "/sessions",
        json={"title": "函数课", "subject": "数学", "gender": "女"},
    )
    session_id = created.json()["id"]
    history = RepositoryChatHistory(session_id, repository)
    history.add_messages([HumanMessage(content="你好"), AIMessage(content="你好，同学")])

    assert created.status_code == 201
    assert client.get("/sessions").json()[0]["id"] == session_id
    assert client.get(f"/sessions/{session_id}").json()["settings"]["subject"] == "数学"
    updated = client.patch(
        f"/sessions/{session_id}", json={"title": "新的函数课", "personality": "耐心"}
    )
    assert updated.json()["title"] == "新的函数课"
    messages = client.get(f"/sessions/{session_id}/messages").json()
    assert [message["content"] for message in messages] == ["你好", "你好，同学"]

    assert client.delete(f"/sessions/{session_id}/messages").status_code == 204
    assert client.get(f"/sessions/{session_id}/messages").json() == []

    deleted = client.delete(f"/sessions/{session_id}")
    assert deleted.status_code == 204
    assert client.get(f"/sessions/{session_id}").status_code == 404


def test_chat_endpoint_reuses_teacher_service(tmp_path, monkeypatch):
    service = FakeChatService()
    client, _repository, _service = build_client(tmp_path, monkeypatch, service)
    created = client.post(
        "/sessions",
        json={"subject": "数学", "gender": "女", "personality": "耐心"},
    ).json()

    response = client.post(
        "/chat",
        json={"session_id": created["id"], "message": "顶点公式是什么？"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "这是测试回答。"
    assert response.json()["tool_names"] == ["search_textbook"]
    assert service.calls[0]["session_id"] == created["id"]
    assert service.calls[0]["subject"] == "数学"


def test_streamlit_api_repository_adapter(tmp_path, monkeypatch):
    client, repository, _service = build_client(tmp_path, monkeypatch)
    api_repository = ApiStorageRepository("http://test")
    api_repository.client.close()
    api_repository.client = client

    created = api_repository.create_session(title="API 会话")
    api_repository.update_session(created.id, settings={"subject": "数学"})
    RepositoryChatHistory(created.id, repository).add_messages(
        [HumanMessage(content="问题"), AIMessage(content="回答")]
    )

    assert api_repository.get_session(created.id).settings["subject"] == "数学"
    assert len(api_repository.load_messages(created.id)) == 2
    api_repository.clear_messages(created.id)
    assert api_repository.load_messages(created.id) == []
