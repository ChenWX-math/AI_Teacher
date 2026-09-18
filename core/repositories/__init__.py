"""会话、消息与教学状态的持久化边界。"""

from core.repositories.base import SessionRecord, StorageRepository
from core.repositories.factory import get_repository

__all__ = ["SessionRecord", "StorageRepository", "get_repository"]
