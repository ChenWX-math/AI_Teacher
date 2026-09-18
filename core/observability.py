"""不依赖外部平台的结构化 JSONL 可观测日志。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from contextvars import ContextVar, Token
from datetime import datetime
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.config import PROJECT_ROOT, get_optional_int

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)

_SECRET_FRAGMENTS = ("api_key", "apikey", "token", "secret", "password", "authorization")
_CONTENT_KEYS = {
    "answer",
    "content",
    "prompt",
    "question",
    "student_answer",
    "user_input",
}


def _content_logging_enabled() -> bool:
    return os.getenv("APP_LOG_USER_CONTENT", "false").strip().lower() in {"1", "true", "yes"}


def _content_fingerprint(value: Any) -> dict[str, Any]:
    text = str(value)
    return {
        "chars": len(text),
        "sha256_12": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
        "redacted": True,
    }


def sanitize_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """移除密钥；学生自由文本默认只记录长度和不可逆短指纹。"""
    sanitized: dict[str, Any] = {}
    for key, value in fields.items():
        lowered = key.lower()
        if any(fragment in lowered for fragment in _SECRET_FRAGMENTS):
            sanitized[key] = "[REDACTED]"
        elif lowered in _CONTENT_KEYS and not _content_logging_enabled():
            sanitized[key] = _content_fingerprint(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_fields(value)
        elif isinstance(value, (list, tuple)):
            sanitized[key] = list(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized


def build_event(event: str, **fields: Any) -> dict[str, Any]:
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "event": event,
        "request_id": fields.pop("request_id", None) or _request_id.get(),
        "session_id": fields.pop("session_id", None) or _session_id.get(),
        **sanitize_fields(fields),
    }
    return {key: value for key, value in payload.items() if value is not None}


def bind_trace(session_id: str) -> tuple[str, Token, Token]:
    request_id = uuid4().hex
    return request_id, _request_id.set(request_id), _session_id.set(session_id)


def reset_trace(request_token: Token, session_token: Token) -> None:
    _request_id.reset(request_token)
    _session_id.reset(session_token)


@lru_cache(maxsize=1)
def _event_logger() -> logging.Logger:
    configured_path = os.getenv("APP_LOG_PATH", "logs/events.jsonl").strip()
    path = Path(configured_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ai_teacher.events")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = RotatingFileHandler(
        path,
        maxBytes=get_optional_int("APP_LOG_MAX_BYTES", 5_242_880, minimum=1024),
        backupCount=get_optional_int("APP_LOG_BACKUP_COUNT", 3, minimum=1),
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def record_event(event: str, **fields: Any) -> None:
    """写入一行 JSON；日志失败不能影响主业务。"""
    try:
        _event_logger().info(json.dumps(build_event(event, **fields), ensure_ascii=False))
    except Exception:
        logging.getLogger(__name__).exception("observability_event_write_failed event=%s", event)
