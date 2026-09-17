"""本地 JSON 会话历史测试。"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from core.memory import (
    JsonChatHistory,
    create_session,
    delete_session,
    list_sessions,
    load_session_meta,
    save_session_meta,
)


def test_chat_history_round_trip(tmp_path):
    history = JsonChatHistory("lesson-1", directory=str(tmp_path))
    history.add_messages([HumanMessage(content="你好"), AIMessage(content="你好，同学")])

    loaded = JsonChatHistory("lesson-1", directory=str(tmp_path)).messages
    assert [message.content for message in loaded] == ["你好", "你好，同学"]

    history.clear()
    assert history.messages == []


def test_session_lifecycle_and_meta(tmp_path):
    directory = str(tmp_path)
    session_id = create_session(directory)
    save_session_meta(session_id, {"subject": "数学"}, directory)

    assert session_id in list_sessions(directory)
    assert load_session_meta(session_id, directory) == {"subject": "数学"}

    delete_session(session_id, directory)
    assert session_id not in list_sessions(directory)
    assert load_session_meta(session_id, directory) == {}


def test_corrupted_history_has_actionable_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="会话文件损坏或无法读取"):
        _ = JsonChatHistory("broken", directory=str(tmp_path)).messages


def test_corrupted_meta_falls_back_to_empty_dict(tmp_path):
    (tmp_path / "lesson.meta.json").write_text(json.dumps(["wrong-shape"]), encoding="utf-8")

    # JSON 有效但结构不符合预期时，调用方仍需拿到字典。
    assert load_session_meta("lesson", str(tmp_path)) == {}
