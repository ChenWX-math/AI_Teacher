"""独立于聊天消息的结构化教学状态及其本地持久化。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from core.memory import SESSIONS_DIR


class ExerciseState(BaseModel):
    topic: str
    difficulty: str
    question_type: str
    question: str
    reference_answer: str
    explanation: str
    created_at: str = Field(default_factory=lambda: datetime.now().astimezone().isoformat())


class GradeState(BaseModel):
    correct: bool
    score: int = Field(ge=0, le=100)
    feedback: str
    error_reason: str
    hint: str
    reference_solution: str
    next_step: str = ""
    graded_at: str = Field(default_factory=lambda: datetime.now().astimezone().isoformat())


class TeachingState(BaseModel):
    schema_version: int = 1
    subject: str | None = None
    current_topic: str | None = None
    current_exercise: ExerciseState | None = None
    last_grade: GradeState | None = None
    conversation_summary: str = ""
    summarized_message_count: int = Field(default=0, ge=0)


def teaching_state_path(session_id: str, directory: str | Path | None = None) -> Path:
    root = Path(directory or SESSIONS_DIR)
    return root / f"{session_id}.state.json"


def load_teaching_state(
    session_id: str, directory: str | Path | None = None
) -> TeachingState:
    path = teaching_state_path(session_id, directory)
    if not path.exists():
        return TeachingState()
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return TeachingState()
        return TeachingState.model_validate(data)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError):
        return TeachingState()


def save_teaching_state(
    session_id: str,
    state: TeachingState,
    directory: str | Path | None = None,
) -> None:
    """原子写入状态，避免进程中断留下半个 JSON 文件。"""
    path = teaching_state_path(session_id, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def clear_teaching_state(session_id: str, directory: str | Path | None = None) -> None:
    path = teaching_state_path(session_id, directory)
    if path.exists():
        path.unlink()


def format_teaching_state_for_agent(state: TeachingState) -> str:
    """只暴露完成当前任务所需信息，不把隐藏参考答案交给主 Agent。"""
    lines = ["【当前教学状态】"]
    lines.append(f"当前科目：{state.subject or '未记录'}")
    lines.append(f"当前知识点：{state.current_topic or '未记录'}")
    if state.current_exercise:
        exercise = state.current_exercise
        lines.extend([
            f"当前练习题：{exercise.question}",
            f"题目难度：{exercise.difficulty}",
            f"题型：{exercise.question_type}",
            "参考答案由批改工具内部保存，主回答不得猜测或提前泄露。",
        ])
    else:
        lines.append("当前练习题：无")
    if state.last_grade:
        lines.append(
            f"最近批改：{state.last_grade.score} 分；错误原因："
            f"{state.last_grade.error_reason or '无'}"
        )
    else:
        lines.append("最近批改：无")
    return "\n".join(lines)
