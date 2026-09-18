"""长对话的分层上下文：摘要、最近消息和结构化教学状态。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from core.config import get_optional_float, get_optional_int
from core.teaching_state import TeachingState

logger = logging.getLogger(__name__)


@dataclass
class PreparedContext:
    messages: list[BaseMessage]
    estimated_tokens: int
    used_summary: bool = False
    summary_updated: bool = False
    fallback_used: bool = False


class LayeredContextManager:
    def __init__(
        self,
        summarizer: Callable[[str], str],
        *,
        max_tokens: int | None = None,
        recent_message_count: int | None = None,
        summary_max_chars: int | None = None,
        chars_per_token: float | None = None,
    ):
        self.summarizer = summarizer
        self.max_tokens = max_tokens or get_optional_int(
            "AGENT_CONTEXT_MAX_TOKENS", 12000, minimum=1000
        )
        self.recent_message_count = recent_message_count or get_optional_int(
            "AGENT_CONTEXT_RECENT_MESSAGES", 8, minimum=2
        )
        self.summary_max_chars = summary_max_chars or get_optional_int(
            "AGENT_SUMMARY_MAX_CHARS", 2000, minimum=200
        )
        self.chars_per_token = chars_per_token or get_optional_float(
            "AGENT_CHARS_PER_TOKEN", 1.5, minimum=0.5
        )

    def count_tokens(self, messages: list[BaseMessage]) -> int:
        return count_tokens_approximately(messages, chars_per_token=self.chars_per_token)

    @staticmethod
    def _summary_message(summary: str) -> SystemMessage:
        return SystemMessage(
            content=(
                "以下是较早对话的压缩摘要。把它作为背景使用；如果与最近原文或结构化"
                f"教学状态冲突，以后两者为准。\n\n{summary}"
            )
        )

    @staticmethod
    def _transcript(messages: list[BaseMessage]) -> str:
        labels = {"human": "学生", "ai": "AI教师", "system": "系统", "tool": "工具"}
        return "\n".join(
            f"{labels.get(message.type, message.type)}：{message.content}" for message in messages
        )

    def _update_summary(self, state: TeachingState, messages: list[BaseMessage]) -> str:
        prompt = f"""请把下面的较早对话压缩成不超过 {self.summary_max_chars} 个中文字符的事实摘要。
必须保留：学生明确偏好、已经讲过的知识点、仍未解决的问题、重要错误原因和双方约定。
不要补充对话中不存在的事实。结构化的当前练习题会由系统单独保存，不必复制完整答案。

已有摘要：
{state.conversation_summary or '无'}

新增较早对话：
{self._transcript(messages)}
"""
        summary = self.summarizer(prompt).strip()
        if not summary:
            raise ValueError("摘要模型返回空内容")
        return summary[: self.summary_max_chars]

    def prepare(
        self,
        history_messages: list[BaseMessage],
        state: TeachingState,
        *,
        pending_messages: list[BaseMessage] | None = None,
    ) -> PreparedContext:
        """准备本轮上下文；必要时更新 state 中的早期对话摘要。"""
        pending_messages = pending_messages or []
        summarized_count = min(state.summarized_message_count, len(history_messages))
        unsummarized = history_messages[summarized_count:]
        base_messages = (
            [self._summary_message(state.conversation_summary)]
            if state.conversation_summary
            else []
        ) + unsummarized

        if self.count_tokens(base_messages + pending_messages) <= self.max_tokens:
            return PreparedContext(
                messages=base_messages,
                estimated_tokens=self.count_tokens(base_messages + pending_messages),
                used_summary=bool(state.conversation_summary),
            )

        keep_count = min(self.recent_message_count, len(unsummarized))
        cutoff = len(history_messages) - keep_count
        new_older_messages = history_messages[summarized_count:cutoff]
        try:
            if new_older_messages:
                state.conversation_summary = self._update_summary(state, new_older_messages)
                state.summarized_message_count = cutoff
            recent_messages = history_messages[cutoff:]
            prepared = [self._summary_message(state.conversation_summary), *recent_messages]
            available_tokens = max(
                1,
                self.max_tokens - self.count_tokens(pending_messages),
            )
            prepared = trim_messages(
                prepared,
                max_tokens=available_tokens,
                token_counter=lambda messages: self.count_tokens(messages),
                strategy="last",
                allow_partial=True,
                include_system=True,
            )
            return PreparedContext(
                messages=prepared,
                estimated_tokens=self.count_tokens(prepared + pending_messages),
                used_summary=True,
                summary_updated=bool(new_older_messages),
            )
        except Exception:
            logger.exception("conversation_summary_failed; using sliding window fallback")
            available_tokens = max(1, self.max_tokens - self.count_tokens(pending_messages))
            fallback = trim_messages(
                history_messages,
                max_tokens=available_tokens,
                token_counter=lambda messages: self.count_tokens(messages),
                strategy="last",
                allow_partial=True,
            )
            return PreparedContext(
                messages=fallback,
                estimated_tokens=self.count_tokens(fallback + pending_messages),
                fallback_used=True,
            )
