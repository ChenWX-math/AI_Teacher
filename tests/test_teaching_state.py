import json

from core.memory import JsonChatHistory, delete_session, list_sessions
from core.teaching_state import (
    ExerciseState,
    TeachingState,
    clear_teaching_state,
    format_teaching_state_for_agent,
    load_teaching_state,
    save_teaching_state,
    teaching_state_path,
)


def build_state(answer="隐藏答案"):
    return TeachingState(
        subject="数学",
        current_topic="二次函数",
        current_exercise=ExerciseState(
            topic="二次函数",
            difficulty="基础",
            question_type="解答题",
            question="求 y=x² 的顶点。",
            reference_answer=answer,
            explanation="使用顶点式。",
        ),
    )


def test_teaching_state_round_trip_and_session_isolation(tmp_path):
    save_teaching_state("session-a", build_state("答案 A"), tmp_path)
    save_teaching_state("session-b", build_state("答案 B"), tmp_path)

    assert load_teaching_state("session-a", tmp_path).current_exercise.reference_answer == "答案 A"
    assert load_teaching_state("session-b", tmp_path).current_exercise.reference_answer == "答案 B"


def test_invalid_state_file_falls_back_to_empty_state(tmp_path):
    teaching_state_path("broken", tmp_path).write_text("{not-json", encoding="utf-8")

    state = load_teaching_state("broken", tmp_path)

    assert state.current_exercise is None
    assert state.conversation_summary == ""


def test_agent_state_context_does_not_leak_reference_answer():
    context = format_teaching_state_for_agent(build_state("不能泄露的答案"))

    assert "求 y=x² 的顶点" in context
    assert "不能泄露的答案" not in context
    assert "批改工具内部保存" in context


def test_state_file_is_not_listed_as_chat_and_is_deleted_with_session(tmp_path):
    history = JsonChatHistory("lesson", directory=str(tmp_path))
    history.clear()
    save_teaching_state("lesson", build_state(), tmp_path)
    (tmp_path / "lesson.meta.json").write_text(json.dumps({"subject": "数学"}), encoding="utf-8")

    assert list_sessions(tmp_path) == ["lesson"]

    delete_session("lesson", tmp_path)

    assert not history.path.endswith(".state.json")
    assert not (tmp_path / "lesson.json").exists()
    assert not (tmp_path / "lesson.meta.json").exists()
    assert not teaching_state_path("lesson", tmp_path).exists()


def test_clear_state_keeps_chat_history_but_removes_current_task(tmp_path):
    history = JsonChatHistory("lesson", directory=str(tmp_path))
    history.add_user_message("保留在测试中的聊天消息")
    save_teaching_state("lesson", build_state(), tmp_path)

    clear_teaching_state("lesson", tmp_path)

    assert len(history.messages) == 1
    assert load_teaching_state("lesson", tmp_path).current_exercise is None
