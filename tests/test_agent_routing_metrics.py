"""Agent 路由评测辅助函数测试。"""

from langchain_core.messages import AIMessage, ToolMessage

from tests.evaluate_agent_routing import (
    extract_tool_calls,
    extract_tool_names,
    percentile_95,
    validate_tool_arguments,
)


def test_extract_tool_names_deduplicates_tool_results():
    messages = [
        AIMessage(content=""),
        ToolMessage(content="a", tool_call_id="1", name="search_textbook"),
        ToolMessage(content="b", tool_call_id="2", name="search_textbook"),
        ToolMessage(content="c", tool_call_id="3", name="grade_answer"),
    ]

    assert extract_tool_names(messages) == ["search_textbook", "grade_answer"]


def test_tool_argument_validation_uses_pydantic_schemas():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "generate_exercise",
                "args": {"topic": "二次函数"},
                "id": "1",
                "type": "tool_call",
            }],
        )
    ]
    calls = extract_tool_calls(messages)

    assert validate_tool_arguments(calls, ["generate_exercise"])
    assert not validate_tool_arguments(calls, ["grade_answer"])


def test_p95_uses_nearest_rank():
    assert percentile_95(list(range(1, 101))) == 95
