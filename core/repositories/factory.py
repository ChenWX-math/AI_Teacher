"""按环境配置选择存储后端。"""

from __future__ import annotations

from functools import lru_cache

from core.config import PROJECT_ROOT, get_storage_backend
from core.repositories.base import StorageRepository
from core.repositories.json_repository import JsonStorageRepository


@lru_cache(maxsize=2)
def _build_repository(backend: str) -> StorageRepository:
    if backend == "json":
        return JsonStorageRepository(PROJECT_ROOT / "sessions")
    if backend == "postgres":
        from core.repositories.postgres_repository import PostgresStorageRepository

        return PostgresStorageRepository.from_config()
    raise ValueError("STORAGE_BACKEND 必须是 json 或 postgres")


def get_repository(backend: str | None = None) -> StorageRepository:
    selected = backend.strip().lower() if backend else get_storage_backend()
    return _build_repository(selected)


def clear_repository_cache() -> None:
    _build_repository.cache_clear()
