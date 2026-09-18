from functools import lru_cache

from core.chat_service import TeacherChatService
from core.repositories.base import StorageRepository
from core.repositories.factory import get_repository


@lru_cache(maxsize=1)
def get_storage_repository() -> StorageRepository:
    return get_repository()


def get_chat_service() -> TeacherChatService:
    return TeacherChatService(get_storage_repository())
