"""Prompt 配置与构建测试。"""

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
