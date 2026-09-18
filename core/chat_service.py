"""可被 Streamlit 与 FastAPI 共同调用的 Teacher Agent 应用服务。"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from core.context_manager import LayeredContextManager
from core.llm import get_llm
from core.memory import get_session_history
from core.rag import KnowledgeBase
from core.repositories.base import StorageRepository
from core.teacher_agent import TeacherAgentResult, build_teacher_agent, run_teacher_agent
from core.teacher_tools import create_teacher_tools
from core.teaching_state import load_teaching_state, save_teaching_state


@lru_cache(maxsize=1)
def load_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase().load()


class TeacherChatService:
    def __init__(
        self,
        repository: StorageRepository,
        *,
        knowledge_base_loader: Callable[[], KnowledgeBase] = load_knowledge_base,
        llm_factory: Callable[..., BaseChatModel] = get_llm,
    ):
        self.repository = repository
        self.knowledge_base_loader = knowledge_base_loader
        self.llm_factory = llm_factory

    def chat(
        self,
        *,
        session_id: str,
        message: str,
        subject: str,
        gender: str,
        personality: str,
        top_k: int = 3,
    ) -> TeacherAgentResult:
        if self.repository.get_session(session_id) is None:
            raise KeyError(session_id)
        if not message.strip():
            raise ValueError("message 不能为空")

        with self.repository.transaction() as transaction:
            state = load_teaching_state(session_id, repository=transaction)

            def save_state(current_state) -> None:
                save_teaching_state(session_id, current_state, repository=transaction)

            tools = create_teacher_tools(
                subject=subject,
                top_k=top_k,
                knowledge_base_loader=self.knowledge_base_loader,
                llm_factory=lambda: self.llm_factory(streaming=False, temperature=0.2),
                teaching_state=state,
                state_saver=save_state,
                strict_state_persistence=True,
            )
            agent = build_teacher_agent(
                model=self.llm_factory(streaming=False),
                tools=tools,
                subject=subject,
                gender=gender,
                personality=personality,
                teaching_state=state,
            )
            context_manager = LayeredContextManager(
                summarizer=lambda prompt: str(
                    getattr(
                        self.llm_factory(streaming=False, temperature=0).invoke(prompt),
                        "content",
                        "",
                    )
                )
            )
            return run_teacher_agent(
                agent,
                user_input=message,
                history=get_session_history(session_id, transaction),
                context_manager=context_manager,
                teaching_state=state,
                state_saver=save_state,
                strict_state_persistence=True,
            )
