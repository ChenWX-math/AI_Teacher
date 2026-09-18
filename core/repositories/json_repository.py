"""兼容现有 sessions/*.json 文件布局的 JSON 存储后端。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.repositories.base import SessionRecord, StorageRepository


def create_session_id() -> str:
    return f"{datetime.now():%Y-%m-%d_%H-%M-%S-%f}_{uuid4().hex[:8]}"


class JsonStorageRepository(StorageRepository):
    """保留聊天、meta 和 state 三类旧 JSON 文件的兼容实现。"""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _messages_path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"

    def _meta_path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.meta.json"

    def _state_path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.state.json"

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_write(path: Path, value: Any, *, indent: int | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temp_path.write_text(
                json.dumps(value, ensure_ascii=False, indent=indent),
                encoding="utf-8",
            )
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def create_session(
        self, *, session_id: str | None = None, title: str | None = None
    ) -> SessionRecord:
        while True:
            candidate = session_id or create_session_id()
            path = self._messages_path(candidate)
            try:
                with path.open("x", encoding="utf-8") as file:
                    json.dump([], file, ensure_ascii=False)
                break
            except FileExistsError:
                if session_id is not None:
                    raise
        if title and title != candidate:
            self.update_session(candidate, title=title)
        record = self.get_session(candidate)
        assert record is not None
        return record

    def list_sessions(self) -> list[SessionRecord]:
        session_ids = sorted(
            (
                path.name[:-5]
                for path in self.directory.glob("*.json")
                if not path.name.endswith((".meta.json", ".state.json"))
            ),
            reverse=True,
        )
        return [record for session_id in session_ids if (record := self.get_session(session_id))]

    def get_session(self, session_id: str) -> SessionRecord | None:
        path = self._messages_path(session_id)
        if not path.exists():
            return None
        meta = self.load_session_meta(session_id)
        stat = path.stat()
        return SessionRecord(
            id=session_id,
            title=str(meta.get("title") or session_id),
            created_at=datetime.fromtimestamp(stat.st_ctime).astimezone(),
            updated_at=datetime.fromtimestamp(stat.st_mtime).astimezone(),
            settings={key: value for key, value in meta.items() if key != "title"},
        )

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> SessionRecord | None:
        if not self._messages_path(session_id).exists():
            return None
        meta = self.load_session_meta(session_id)
        if title is not None:
            meta["title"] = title
        if settings is not None:
            meta.update(settings)
        self.save_session_meta(session_id, meta)
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        for path in (
            self._messages_path(session_id),
            self._meta_path(session_id),
            self._state_path(session_id),
        ):
            if path.exists():
                path.unlink()

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        path = self._messages_path(session_id)
        if not path.exists():
            return []
        data = self._read_json(path)
        if not isinstance(data, list):
            raise TypeError("聊天历史根节点必须是数组")
        return data

    def replace_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        self._atomic_write(self._messages_path(session_id), messages, indent=2)

    def load_session_meta(self, session_id: str) -> dict[str, Any]:
        path = self._meta_path(session_id)
        if not path.exists():
            return {}
        try:
            data = self._read_json(path)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError, TypeError):
            return {}

    def save_session_meta(self, session_id: str, meta: dict[str, Any]) -> None:
        self._atomic_write(self._meta_path(session_id), meta)

    def load_teaching_state(self, session_id: str) -> dict[str, Any] | None:
        path = self._state_path(session_id)
        if not path.exists():
            return None
        data = self._read_json(path)
        return data if isinstance(data, dict) else None

    def save_teaching_state(self, session_id: str, state: dict[str, Any]) -> None:
        self._atomic_write(self._state_path(session_id), state, indent=2)

    def clear_teaching_state(self, session_id: str) -> None:
        path = self._state_path(session_id)
        if path.exists():
            path.unlink()

    def teaching_state_path(self, session_id: str) -> Path:
        return self._state_path(session_id)
