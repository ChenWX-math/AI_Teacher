"""Agent 结果解析与历史持久化测试。"""

from langchain_core.messages import AIMessage, ToolMessage

from core.memory import JsonChatHistory
from core.teacher_agent import run_teacher_agent


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
