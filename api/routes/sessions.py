from fastapi import APIRouter, Depends, HTTPException, Response, status

from api.dependencies import get_storage_repository
from api.schemas import MessageResponse, SessionCreate, SessionResponse, SessionUpdate
from core.repositories.base import SessionRecord, StorageRepository

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _session_response(record: SessionRecord) -> SessionResponse:
    return SessionResponse(
        id=record.id,
        title=record.title,
        created_at=record.created_at,
        updated_at=record.updated_at,
        settings=record.settings,
    )


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: SessionCreate,
    repository: StorageRepository = Depends(get_storage_repository),
):
    settings = {
        key: value
        for key, value in {
            "subject": payload.subject,
            "gender": payload.gender,
            "personality": payload.personality,
        }.items()
        if value is not None
    }
    record = repository.create_session(title=payload.title)
    if settings:
        record = repository.update_session(record.id, settings=settings) or record
    return _session_response(record)


@router.get("", response_model=list[SessionResponse])
def list_sessions(repository: StorageRepository = Depends(get_storage_repository)):
    return [_session_response(record) for record in repository.list_sessions()]


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: str,
    repository: StorageRepository = Depends(get_storage_repository),
):
    record = repository.get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="session not found")
    return _session_response(record)


@router.patch("/{session_id}", response_model=SessionResponse)
def update_session(
    session_id: str,
    payload: SessionUpdate,
    repository: StorageRepository = Depends(get_storage_repository),
):
    settings = {
        key: value
        for key, value in {
            "subject": payload.subject,
            "gender": payload.gender,
            "personality": payload.personality,
        }.items()
        if value is not None
    }
    record = repository.update_session(
        session_id,
        title=payload.title,
        settings=settings or None,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="session not found")
    return _session_response(record)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: str,
    repository: StorageRepository = Depends(get_storage_repository),
):
    if repository.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="session not found")
    repository.delete_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{session_id}/messages", response_model=list[MessageResponse])
def get_messages(
    session_id: str,
    repository: StorageRepository = Depends(get_storage_repository),
):
    if repository.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="session not found")
    responses = []
    for payload in repository.load_messages(session_id):
        data = payload.get("data", {})
        responses.append(
            MessageResponse(
                role=str(payload.get("type", data.get("type", "unknown"))),
                content=str(data.get("content", "")),
                metadata=dict(data.get("response_metadata", {})),
            )
        )
    return responses


@router.delete("/{session_id}/messages", status_code=status.HTTP_204_NO_CONTENT)
def clear_messages(
    session_id: str,
    repository: StorageRepository = Depends(get_storage_repository),
):
    if repository.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="session not found")
    with repository.transaction() as transaction:
        transaction.clear_messages(session_id)
        transaction.clear_teaching_state(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
