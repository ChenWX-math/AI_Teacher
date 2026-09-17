"""
LangChain 记忆（会话持久化）模块

用自定义的 JsonChatHistory 实现 BaseChatMessageHistory：把每个会话的消息历史
持久化到 sessions/{session_id}.json。

LangChain 1.x 的 BaseChatMessageHistory 约定：
    - 需要实现 messages 属性（读）、add_messages（批量写）、clear（清空）
    - add_message（单条）会默认委托给 add_messages，不用重复实现
"""

import json
import os
from datetime import datetime
from uuid import uuid4

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import messages_from_dict, messages_to_dict

# sessions 目录固定放在项目根目录（AI_Teacher/）下
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(_PROJECT_ROOT, "sessions")


class JsonChatHistory(BaseChatMessageHistory):
    """把消息历史持久化到本地 JSON 文件的历史存储。"""

    def __init__(self, session_id, directory=None):
        """初始化一个会话历史。

        Args:
            session_id (str): 会话 ID，对应文件 sessions/{session_id}.json。
            directory (str, optional): 存储目录，默认 SESSIONS_DIR。
        """
        self.session_id = session_id
        self.directory = directory or SESSIONS_DIR
        os.makedirs(self.directory, exist_ok=True)
        self._path = os.path.join(self.directory, f"{session_id}.json")

    @property
    def messages(self):
        """读取当前会话的全部消息（文件不存在时返回空列表）。"""
        if not os.path.exists(self._path):
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return messages_from_dict(data)
        except (json.JSONDecodeError, OSError, TypeError, KeyError) as exc:
            raise ValueError(f"会话文件损坏或无法读取：{self._path}（{exc}）") from exc

    def add_messages(self, messages):
        """批量追加消息并写回文件（读旧 + 追加新，再整体落盘）。

        Args:
            messages (Sequence[BaseMessage]): 要追加的消息列表。
        """
        all_messages = list(self.messages) + list(messages)
        with open(self._path, "w", encoding="utf-8") as f:
            # ensure_ascii=False：中文不转义成 \uXXXX
            json.dump(messages_to_dict(all_messages), f, ensure_ascii=False, indent=2)

    def clear(self):
        """清空当前会话的所有消息。"""
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False)

    @property
    def path(self):
        return self._path


def create_session_id():
    """仅生成 ID；随机后缀避免快速点击或多个页面创建时重名。"""
    return f"{datetime.now():%Y-%m-%d_%H-%M-%S-%f}_{uuid4().hex[:8]}"


def create_session(directory=None):
    """创建空会话文件并返回 ID，让侧边栏能立即列出新会话。"""
    while True:
        session_id = create_session_id()
        history = JsonChatHistory(session_id, directory=directory)
        try:
            # x 模式只创建新文件；即使 ID 意外重复，也不会覆盖已有历史。
            with open(history.path, "x", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False)
        except FileExistsError:
            continue
        return session_id


def list_sessions(directory=None):
    directory = directory or SESSIONS_DIR
    os.makedirs(directory, exist_ok=True)
    return sorted(
        (
            name[:-5]
            for name in os.listdir(directory)
            if name.endswith(".json") and not name.endswith(".meta.json")
        ),
        reverse=True,
    )


def delete_session(session_id, directory=None):
    directory = directory or SESSIONS_DIR
    history = JsonChatHistory(session_id, directory=directory)
    if os.path.exists(history.path):
        os.remove(history.path)
    meta_path = os.path.join(directory, f"{session_id}.meta.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)


def save_session_meta(session_id, meta, directory=None):
    """把科目/性别/性格等设置持久化到 {session_id}.meta.json。"""
    directory = directory or SESSIONS_DIR
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{session_id}.meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def load_session_meta(session_id, directory=None):
    """读取 {session_id}.meta.json；文件不存在或损坏时返回空字典。"""
    directory = directory or SESSIONS_DIR
    path = os.path.join(directory, f"{session_id}.meta.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def get_session_history(session_id):
    """工厂函数：返回指定 session_id 对应的 JsonChatHistory 实例。

    供 RunnableWithMessageHistory(get_session_history=...) 使用。
    """
    return JsonChatHistory(session_id)
