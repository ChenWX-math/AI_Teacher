"""LangChain 会话历史兼容层。

旧调用仍可使用 ``JsonChatHistory`` 和模块级 CRUD 函数；默认实现改为通过
Repository 接口选择 JSON 或 PostgreSQL 后端。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import messages_from_dict, messages_to_dict

from core.config import PROJECT_ROOT
from core.repositories.base import StorageRepository
from core.repositories.factory import get_repository
from core.repositories.json_repository import JsonStorageRepository, create_session_id

SESSIONS_DIR = os.fspath(PROJECT_ROOT / "sessions")


def _repository(
    directory: str | Path | None = None,
    repository: StorageRepository | None = None,
) -> StorageRepository:
    if repository is not None:
        return repository
    if directory is not None:
        return JsonStorageRepository(directory)
    return get_repository()


class RepositoryChatHistory(BaseChatMessageHistory):
    """把 LangChain 消息序列持久化到选定的 Repository。"""

    def __init__(self, session_id: str, repository: StorageRepository):
        self.session_id = session_id
        self.repository = repository

    @property
    def messages(self):
        try:
            return messages_from_dict(self.repository.load_messages(self.session_id))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"会话文件损坏或无法读取：{self.session_id}（{exc}）") from exc

    def add_messages(self, messages) -> None:
        self.repository.append_messages(self.session_id, messages_to_dict(list(messages)))

    def clear(self) -> None:
        self.repository.clear_messages(self.session_id)


class JsonChatHistory(RepositoryChatHistory):
    """保留原公开类名，并显式使用 JSON 后端。"""

    def __init__(self, session_id: str, directory: str | Path | None = None):
        json_repository = JsonStorageRepository(directory or SESSIONS_DIR)
        super().__init__(session_id, json_repository)

    @property
    def path(self) -> str:
        return os.fspath(self.repository._messages_path(self.session_id))  # type: ignore[attr-defined]


def create_session(
    directory: str | Path | None = None,
    *,
    repository: StorageRepository | None = None,
    title: str | None = None,
) -> str:
    return _repository(directory, repository).create_session(title=title).id


def list_sessions(
    directory: str | Path | None = None,
    *,
    repository: StorageRepository | None = None,
) -> list[str]:
    return [record.id for record in _repository(directory, repository).list_sessions()]


def delete_session(
    session_id: str,
    directory: str | Path | None = None,
    *,
    repository: StorageRepository | None = None,
) -> None:
    _repository(directory, repository).delete_session(session_id)


def save_session_meta(
    session_id: str,
    meta: dict[str, Any],
    directory: str | Path | None = None,
    *,
    repository: StorageRepository | None = None,
) -> None:
    repo = _repository(directory, repository)
    if isinstance(repo, JsonStorageRepository):
        repo.save_session_meta(session_id, meta)
    else:
        repo.update_session(session_id, settings=meta)


def load_session_meta(
    session_id: str,
    directory: str | Path | None = None,
    *,
    repository: StorageRepository | None = None,
) -> dict[str, Any]:
    repo = _repository(directory, repository)
    if isinstance(repo, JsonStorageRepository):
        return repo.load_session_meta(session_id)
    record = repo.get_session(session_id)
    return dict(record.settings) if record else {}


def get_session_history(
    session_id: str,
    repository: StorageRepository | None = None,
) -> RepositoryChatHistory:
    return RepositoryChatHistory(session_id, repository or get_repository())


__all__ = [
    "JsonChatHistory",
    "RepositoryChatHistory",
    "SESSIONS_DIR",
    "create_session",
    "create_session_id",
    "delete_session",
    "get_session_history",
    "list_sessions",
    "load_session_meta",
    "save_session_meta",
]
