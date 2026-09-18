"""存储后端共同遵循的最小接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator


@dataclass(frozen=True)
class SessionRecord:
    id: str
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    settings: dict[str, Any] = field(default_factory=dict)


class StorageRepository(ABC):
    """同时覆盖会话、消息和教学状态，便于一次对话共享事务。"""

    @contextmanager
    def transaction(self) -> Iterator["StorageRepository"]:
        """返回参与同一事务的仓储；JSON 后端使用原子文件写入。"""
        yield self

    def healthcheck(self) -> bool:
        return True

    @abstractmethod
    def create_session(
        self, *, session_id: str | None = None, title: str | None = None
    ) -> SessionRecord:
        raise NotImplementedError

    @abstractmethod
    def list_sessions(self) -> list[SessionRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_session(self, session_id: str) -> SessionRecord | None:
        raise NotImplementedError

    @abstractmethod
    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> SessionRecord | None:
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def replace_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def append_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        self.replace_messages(session_id, [*self.load_messages(session_id), *messages])

    def clear_messages(self, session_id: str) -> None:
        self.replace_messages(session_id, [])

    @abstractmethod
    def load_teaching_state(self, session_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def save_teaching_state(self, session_id: str, state: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear_teaching_state(self, session_id: str) -> None:
        raise NotImplementedError
