"""Prompt 配置与构建测试。"""

from core.prompt_builder import build_chat_prompt
from prompts.prompt_manager import PromptManager


def test_all_expected_subjects_are_available():
    subjects = set(PromptManager().list_subjects())

    assert subjects == {"语文", "数学", "英语", "物理", "化学", "生物", "政治", "历史", "地理"}


def test_system_prompt_contains_persona_and_subject_guidance():
    prompt = PromptManager().build_system_prompt("数学", "女", "耐心、善于举例")

    assert "数学" in prompt
    assert "女" in prompt
    assert "耐心、善于举例" in prompt
    assert "审题" in prompt


def test_rag_prompt_requires_visible_citations():
    prompt = build_chat_prompt("数学", "女", "耐心", use_context=True)

    rendered = prompt.invoke({"history": [], "input": "问题", "context": "教材内容"})
    system_text = "\n".join(str(message.content) for message in rendered.messages)
    assert "[资料 N]" in system_text
    assert "不要编造内容或来源" in system_text
