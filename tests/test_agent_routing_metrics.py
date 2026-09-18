"""Agent 路由评测辅助函数测试。"""

from langchain_core.messages import AIMessage, ToolMessage

from tests.evaluate_agent_routing import extract_tool_names


def test_extract_tool_names_deduplicates_tool_results():
    messages = [
        AIMessage(content=""),
        ToolMessage(content="a", tool_call_id="1", name="search_textbook"),
        ToolMessage(content="b", tool_call_id="2", name="search_textbook"),
        ToolMessage(content="c", tool_call_id="3", name="grade_answer"),
    ]

    assert extract_tool_names(messages) == ["search_textbook", "grade_answer"]
