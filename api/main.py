from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request

from api.routes import chat, health, sessions
from core.observability import record_event

app = FastAPI(title="AI 智能教师 API", version="1.0.0")
app.include_router(health.router)
app.include_router(sessions.router)
app.include_router(chat.router)


@app.middleware("http")
async def observe_request(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    started_at = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        record_event(
            "api.request.completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=status_code,
            latency_ms=round((perf_counter() - started_at) * 1000),
        )
