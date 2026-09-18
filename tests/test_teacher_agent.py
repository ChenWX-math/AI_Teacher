"""Agent 结果解析与历史持久化测试。"""

from langchain_core.messages import AIMessage, ToolMessage

from core.context_manager import LayeredContextManager
from core.memory import JsonChatHistory
from core.teacher_agent import run_teacher_agent
from core.teaching_state import TeachingState


class FakeAgent:
    def invoke(self, inputs, config):
        assert config["recursion_limit"] == 6
        messages = list(inputs["messages"])
        messages.extend([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_textbook",
                    "args": {"query": "顶点"},
                    "id": "call-1",
                    "type": "tool_call",
                }],
            ),
            ToolMessage(
                content="教材内容",
                tool_call_id="call-1",
                name="search_textbook",
                artifact={
                    "sources": [{
                        "label": "资料 1",
                        "source": "数学/函数.txt",
                        "chunk_index": 0,
                        "score": 0.9,
                        "preview": "顶点公式",
                    }]
                },
            ),
            AIMessage(content="顶点公式见 [资料 1]。"),
        ])
        return {"messages": messages}


def test_agent_result_preserves_only_user_and_final_answer(tmp_path):
    history = JsonChatHistory("agent-session", directory=str(tmp_path))

    result = run_teacher_agent(
        FakeAgent(),
        user_input="顶点公式是什么？",
        history=history,
        recursion_limit=6,
    )

    assert result.answer == "顶点公式见 [资料 1]。"
    assert result.tool_names == ["search_textbook"]
    assert result.sources[0]["source"] == "数学/函数.txt"
    assert result.duration_ms >= 0
    assert [message.type for message in history.messages] == ["human", "ai"]


def test_agent_uses_layered_context_and_persists_updated_summary(tmp_path):
    history = JsonChatHistory("long-session", directory=str(tmp_path))
    for index in range(5):
        history.add_user_message(f"第 {index} 个问题" + "很长" * 30)
        history.add_ai_message(f"第 {index} 个回答" + "讲解" * 30)
    state = TeachingState()
    saved = []
    manager = LayeredContextManager(
        lambda _prompt: "较早对话摘要",
        max_tokens=140,
        recent_message_count=2,
        chars_per_token=1,
    )

    result = run_teacher_agent(
        FakeAgent(),
        user_input="顶点公式是什么？",
        history=history,
        recursion_limit=6,
        context_manager=manager,
        teaching_state=state,
        state_saver=lambda current: saved.append(current.model_copy(deep=True)),
    )

    assert result.used_summary
    assert result.context_tokens <= manager.max_tokens
    assert state.conversation_summary == "较早对话摘要"
    assert saved[-1].conversation_summary == "较早对话摘要"
    assert len(history.messages) == 12


def test_summary_state_save_failure_does_not_block_agent_answer(tmp_path):
    history = JsonChatHistory("save-failure", directory=str(tmp_path))
    for index in range(4):
        history.add_user_message(f"问题 {index}" + "很长" * 30)
        history.add_ai_message(f"回答 {index}" + "讲解" * 30)
    manager = LayeredContextManager(
        lambda _prompt: "较早对话摘要",
        max_tokens=140,
        recent_message_count=2,
        chars_per_token=1,
    )

    def fail_to_save(_state):
        raise OSError("disk full")

    result = run_teacher_agent(
        FakeAgent(),
        user_input="顶点公式是什么？",
        history=history,
        recursion_limit=6,
        context_manager=manager,
        teaching_state=TeachingState(),
        state_saver=fail_to_save,
    )

    assert "顶点公式" in result.answer
    assert result.tool_names == ["search_textbook"]
    assert len(history.messages) == 10
