from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_storage_repository
from core.config import get_storage_backend
from core.repositories.base import StorageRepository

router = APIRouter(tags=["health"])


@router.get("/health")
def health(repository: StorageRepository = Depends(get_storage_repository)):
    try:
        healthy = repository.healthcheck()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="storage unavailable") from exc
    if not healthy:
        raise HTTPException(status_code=503, detail="storage unavailable")
    return {"status": "ok", "storage_backend": get_storage_backend()}
