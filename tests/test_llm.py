"""LLM 配置测试，不发起网络请求。"""

import core.llm as llm_module


def test_llm_uses_environment_configuration(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")
    monkeypatch.setenv("DEEPSEEK_TEMPERATURE", "0.2")
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("DEEPSEEK_MAX_RETRIES", "4")

    result = llm_module.get_llm()

    assert isinstance(result, FakeChatOpenAI)
    assert captured["model"] == "test-model"
    assert captured["temperature"] == 0.2
    assert captured["timeout"] == 15.0
    assert captured["max_retries"] == 4
    assert captured["streaming"] is True


def test_llm_allows_non_streaming_temperature_override(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    llm_module.get_llm(streaming=False, temperature=0.1)

    assert captured["streaming"] is False
    assert captured["temperature"] == 0.1
