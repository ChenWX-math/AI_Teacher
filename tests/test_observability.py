import json

from core import observability


def test_sensitive_and_student_content_are_redacted(monkeypatch):
    monkeypatch.setenv("APP_LOG_USER_CONTENT", "false")

    event = observability.build_event(
        "agent.started",
        api_key="secret-value",
        user_input="我的身份证号不应写入日志",
        tool_names=["search_textbook"],
    )

    assert event["api_key"] == "[REDACTED]"
    assert event["user_input"]["redacted"] is True
    assert event["user_input"]["chars"] > 0
    assert "身份证" not in json.dumps(event, ensure_ascii=False)
    assert event["tool_names"] == ["search_textbook"]


def test_record_event_writes_valid_jsonl(monkeypatch, tmp_path):
    log_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("APP_LOG_PATH", str(log_path))
    observability._event_logger.cache_clear()

    observability.record_event("agent.completed", success=True, duration_ms=12)

    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["event"] == "agent.completed"
    assert payload["success"] is True
    assert payload["duration_ms"] == 12
    observability._event_logger.cache_clear()
