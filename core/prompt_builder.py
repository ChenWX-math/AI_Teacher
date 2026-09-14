"""
提示词构建模块

把 prompts/ 下 PromptManager 生成的中文系统提示词，包装成 LangChain 的
ChatPromptTemplate，供 `prompt | llm` 链使用。

模板结构（三段）：
    system   —— 由「科目 + 性别 + 性格」生成的人设/教学提示词
    history  —— 历史消息，由 RunnableWithMessageHistory 自动填充
    human    —— 用户当前输入 {input}
"""

import os
import sys

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 把项目根目录（AI_application/）加入 sys.path，确保能导入 prompts 包。
# prompts/ 目录没有 __init__.py，这里以「命名空间包」的方式导入。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from prompts.prompt_manager import PromptManager  # noqa: E402


def build_chat_prompt(subject, gender, personality, use_context=False):
    """根据三个变量，返回一个带 history 占位符的 ChatPromptTemplate。

    Args:
        subject (str): 科目，如 "数学"。
        gender (str): 性别，如 "男" / "女"。
        personality (str): 性格/教学风格，如 "温柔耐心、擅长用生活例子讲解"。

    Returns:
        ChatPromptTemplate: 含 system / history / human 三段的提示词模板。
    """
    # 1. 复用提示词库，生成系统提示词文本
    pm = PromptManager()
    system_text = pm.build_system_prompt(subject, gender, personality)

    # 2. 包装成 LangChain 的 ChatPromptTemplate
    messages = [("system", system_text)]
    if use_context:
        messages.append((
            "system",
            "以下是从教材知识库检索到的参考资料。请优先依据资料回答；如果资料不足，明确说明，不要编造。\n\n{context}",
        ))
    messages.extend([
        MessagesPlaceholder(variable_name="history"), # type: ignore
        ("human", "{input}"),
    ])
    return ChatPromptTemplate.from_messages(messages)
