from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_chat_service, get_storage_repository
from api.schemas import ChatRequest, ChatResponse
from core.chat_service import TeacherChatService
from core.repositories.base import StorageRepository

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    repository: StorageRepository = Depends(get_storage_repository),
    service: TeacherChatService = Depends(get_chat_service),
):
    record = repository.get_session(payload.session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="session not found")
    settings = dict(record.settings)
    selected = {
        "subject": payload.subject or settings.get("subject") or "数学",
        "gender": payload.gender or settings.get("gender") or "女",
        "personality": payload.personality or settings.get("personality") or "耐心、讲解清晰",
    }
    repository.update_session(payload.session_id, settings=selected)
    try:
        result = service.chat(
            session_id=payload.session_id,
            message=payload.message,
            top_k=payload.top_k,
            **selected,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ChatResponse(
        answer=result.answer,
        tool_names=result.tool_names,
        sources=result.sources,
        duration_ms=result.duration_ms,
        context_tokens=result.context_tokens,
        used_summary=result.used_summary,
        context_fallback_used=result.context_fallback_used,
    )
