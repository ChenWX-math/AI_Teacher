"""显式启用的真实 API 冒烟测试；CI 和默认 pytest 均不会运行。"""

import os

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from core.llm import get_llm
from core.rag import KnowledgeBase
from core.teacher_agent import build_teacher_agent

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="设置 RUN_INTEGRATION_TESTS=1 后才运行真实 API 测试",
)
def test_config_to_rag_and_agent_smoke():
    """验证真实配置、Milvus 检索和真实模型 Agent 路由，但不发送教材正文给 LLM。"""
    retrieved = KnowledgeBase().load().search_with_scores(
        "二次函数的顶点公式", subject="数学", k=1
    )
    assert retrieved
    assert retrieved[0][0].metadata["subject"] == "数学"

    @tool
    def search_textbook(query: str) -> str:
        """检索数学教材中的概念、定义、公式和解题方法。"""
        return f"[资料 1] 合成检索结果：{query}"

    agent = build_teacher_agent(
        model=get_llm(streaming=False, temperature=0),
        tools=[search_textbook],
        subject="数学",
        gender="女",
        personality="耐心简洁",
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content="二次函数顶点公式是什么？")]},
        config={"recursion_limit": 6},
    )

    assert any(
        isinstance(message, ToolMessage) and message.name == "search_textbook"
        for message in result["messages"]
    )
