from langchain_core.messages import AIMessage, HumanMessage

from core.context_manager import LayeredContextManager
from core.teaching_state import TeachingState


def make_history(turns=6):
    messages = []
    for index in range(turns):
        messages.extend([
            HumanMessage(content=f"第 {index} 轮学生问题：" + "函数" * 20),
            AIMessage(content=f"第 {index} 轮教师回答：" + "讲解" * 20),
        ])
    return messages


def test_short_context_does_not_call_summarizer():
    calls = []
    manager = LayeredContextManager(
        lambda prompt: calls.append(prompt) or "摘要",
        max_tokens=1000,
        chars_per_token=1,
    )
    history = make_history(1)

    prepared = manager.prepare(history, TeachingState())

    assert prepared.messages == history
    assert not prepared.used_summary
    assert calls == []


def test_long_context_updates_summary_and_keeps_recent_raw_messages():
    manager = LayeredContextManager(
        lambda _prompt: "学生正在学习函数；早期问题已经解决。",
        max_tokens=180,
        recent_message_count=2,
        summary_max_chars=200,
        chars_per_token=1,
    )
    history = make_history(5)
    state = TeachingState()

    prepared = manager.prepare(history, state)

    assert prepared.used_summary
    assert prepared.summary_updated
    assert state.conversation_summary.startswith("学生正在学习函数")
    assert state.summarized_message_count == len(history) - 2
    assert history[-1].content in [message.content for message in prepared.messages]
    assert prepared.estimated_tokens <= manager.max_tokens


def test_existing_summary_only_processes_new_unsummarized_older_messages():
    prompts = []
    manager = LayeredContextManager(
        lambda prompt: prompts.append(prompt) or "合并后的摘要",
        max_tokens=150,
        recent_message_count=2,
        chars_per_token=1,
    )
    history = make_history(6)
    state = TeachingState(conversation_summary="已有摘要", summarized_message_count=4)

    manager.prepare(history, state)

    assert len(prompts) == 1
    assert "已有摘要" in prompts[0]
    assert "第 0 轮学生问题" not in prompts[0]
    assert state.summarized_message_count == len(history) - 2


def test_summary_failure_uses_bounded_sliding_window():
    def fail(_prompt):
        raise RuntimeError("summary unavailable")

    manager = LayeredContextManager(
        fail,
        max_tokens=150,
        recent_message_count=2,
        chars_per_token=1,
    )

    prepared = manager.prepare(make_history(5), TeachingState())

    assert prepared.fallback_used
    assert not prepared.summary_updated
    assert prepared.estimated_tokens <= manager.max_tokens
