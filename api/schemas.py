from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    title: str | None = None
    subject: str | None = None
    gender: str | None = None
    personality: str | None = None


class SessionUpdate(BaseModel):
    title: str | None = None
    subject: str | None = None
    gender: str | None = None
    personality: str | None = None


class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)
    subject: str | None = None
    gender: str | None = None
    personality: str | None = None
    top_k: int = Field(default=3, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    tool_names: list[str] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    duration_ms: int = 0
    context_tokens: int = 0
    used_summary: bool = False
    context_fallback_used: bool = False
