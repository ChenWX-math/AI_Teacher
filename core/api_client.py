"""Streamlit API 模式使用的 HTTP 客户端与 Repository 适配器。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict

from core.config import get_optional
from core.repositories.base import SessionRecord, StorageRepository


class ApiStorageRepository(StorageRepository):
    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or get_optional("API_BASE_URL", "http://127.0.0.1:8000")).rstrip(
            "/"
        )
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    @staticmethod
    def _record(payload: dict[str, Any]) -> SessionRecord:
        return SessionRecord(
            id=payload["id"],
            title=payload["title"],
            created_at=(
                datetime.fromisoformat(payload["created_at"]) if payload["created_at"] else None
            ),
            updated_at=(
                datetime.fromisoformat(payload["updated_at"]) if payload["updated_at"] else None
            ),
            settings=dict(payload.get("settings", {})),
        )

    @staticmethod
    def _raise(response: httpx.Response) -> None:
        response.raise_for_status()

    def healthcheck(self) -> bool:
        response = self.client.get("/health")
        self._raise(response)
        return response.json().get("status") == "ok"

    def create_session(
        self, *, session_id: str | None = None, title: str | None = None
    ) -> SessionRecord:
        if session_id is not None:
            raise ValueError("API 模式由服务端生成 session_id")
        response = self.client.post("/sessions", json={"title": title})
        self._raise(response)
        return self._record(response.json())

    def list_sessions(self) -> list[SessionRecord]:
        response = self.client.get("/sessions")
        self._raise(response)
        return [self._record(item) for item in response.json()]

    def get_session(self, session_id: str) -> SessionRecord | None:
        response = self.client.get(f"/sessions/{session_id}")
        if response.status_code == 404:
            return None
        self._raise(response)
        return self._record(response.json())

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> SessionRecord | None:
        payload = {**(settings or {})}
        if title is not None:
            payload["title"] = title
        response = self.client.patch(f"/sessions/{session_id}", json=payload)
        if response.status_code == 404:
            return None
        self._raise(response)
        return self._record(response.json())

    def delete_session(self, session_id: str) -> None:
        response = self.client.delete(f"/sessions/{session_id}")
        if response.status_code not in {204, 404}:
            self._raise(response)

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        response = self.client.get(f"/sessions/{session_id}/messages")
        self._raise(response)
        messages = []
        for item in response.json():
            message_class = HumanMessage if item["role"] == "human" else AIMessage
            messages.append(message_class(content=item["content"]))
        return messages_to_dict(messages)

    def replace_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        if messages:
            raise NotImplementedError("API 模式不允许客户端直接替换消息")
        self.clear_messages(session_id)

    def append_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        raise NotImplementedError("API 模式由 /chat 保存消息")

    def clear_messages(self, session_id: str) -> None:
        response = self.client.delete(f"/sessions/{session_id}/messages")
        self._raise(response)

    def load_teaching_state(self, session_id: str) -> dict[str, Any] | None:
        raise NotImplementedError("API 客户端不直接读取隐藏教学状态")

    def save_teaching_state(self, session_id: str, state: dict[str, Any]) -> None:
        raise NotImplementedError("API 客户端不直接写入隐藏教学状态")

    def clear_teaching_state(self, session_id: str) -> None:
        # DELETE messages 已在服务端同时清理教学状态。
        return None

    def chat(self, **payload: Any) -> dict[str, Any]:
        response = self.client.post("/chat", json=payload)
        self._raise(response)
        return response.json()
