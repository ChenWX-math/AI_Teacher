"""轻量 AI 教师 Agent：工具路由、执行结果解析和会话持久化。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from core.config import get_optional_int
from core.memory import JsonChatHistory
from prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


@dataclass
class TeacherAgentResult:
    answer: str
    tool_names: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: int = 0


def build_teacher_agent(
    *,
    model: BaseChatModel,
    tools: list[BaseTool],
    subject: str,
    gender: str,
    personality: str,
):
    persona = PromptManager().build_system_prompt(subject, gender, personality)
    agent_rules = """

【工具使用规则】
1. 先按以下优先级判断：批改答案 > 生成练习 > 教材知识问答 > 普通交流。
2. 只要学生提交了答案、计算结果或解题过程，必须调用 grade_answer，不要自行批改。
3. 只要学生要求出题、练习、测试或调整题目难度，必须调用 generate_exercise。
4. 除上述两类外，只要问题涉及当前学科的概念、公式、定义、解题方法或事实，
   即使你知道答案，也必须先调用 search_textbook，再根据工具结果回答。
5. 只有寒暄、情绪支持、学习计划等不涉及具体学科知识的请求可以直接回答。
6. 不要声称调用了实际没有调用的工具；一次请求只调用完成任务所必需的工具。
7. 使用教材工具后，用 [资料 N] 标注依据；资料不足时明确说明。
8. 工具返回错误时，简要说明并给出可继续操作的建议，不要反复调用同一失败工具。
"""
    return create_agent(model=model, tools=tools, system_prompt=persona + agent_rules)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in content
        )
    return str(content or "")


def run_teacher_agent(
    agent,
    *,
    user_input: str,
    history: JsonChatHistory,
    recursion_limit: int | None = None,
) -> TeacherAgentResult:
    """执行一次 Agent，并只把用户消息和最终回答写入现有聊天历史。"""
    if not user_input.strip():
        raise ValueError("user_input 不能为空")
    limit = recursion_limit or get_optional_int("AGENT_RECURSION_LIMIT", 8, minimum=2)
    previous_messages = list(history.messages)
    input_messages = [*previous_messages, HumanMessage(content=user_input)]
    started_at = perf_counter()
    try:
        result = agent.invoke(
            {"messages": input_messages},
            config={"recursion_limit": limit},
        )
    except Exception:
        duration_ms = round((perf_counter() - started_at) * 1000)
        logger.exception(
            "agent_run_failed session_id=%s duration_ms=%s",
            history.session_id,
            duration_ms,
        )
        raise
    messages: list[BaseMessage] = list(result.get("messages", []))
    generated_messages = messages[len(input_messages):]

    answer = ""
    tool_names = []
    sources = []
    artifacts = []
    for message in generated_messages:
        if isinstance(message, ToolMessage):
            if message.name and message.name not in tool_names:
                tool_names.append(message.name)
            if isinstance(message.artifact, dict):
                artifacts.append(message.artifact)
                sources.extend(message.artifact.get("sources", []))
        elif isinstance(message, AIMessage) and not message.tool_calls:
            text = _content_to_text(message.content)
            if text:
                answer = text

    if not answer:
        raise RuntimeError("Agent 没有生成最终回答")

    duration_ms = round((perf_counter() - started_at) * 1000)
    history.add_messages([HumanMessage(content=user_input), AIMessage(content=answer)])
    logger.info(
        "agent_run_completed session_id=%s tools=%s duration_ms=%s",
        history.session_id,
        ",".join(tool_names) or "none",
        duration_ms,
    )
    return TeacherAgentResult(
        answer=answer,
        tool_names=tool_names,
        sources=sources,
        artifacts=artifacts,
        duration_ms=duration_ms,
    )
