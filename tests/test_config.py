"""不访问外部服务的配置单元测试。"""

import pytest

from core.config import get_milvus_uri, get_optional_float, get_optional_int


def test_remote_milvus_uri_does_not_require_local_path(monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "https://milvus.example.com")
    monkeypatch.delenv("MILVUS_DB_PATH", raising=False)

    assert get_milvus_uri() == "https://milvus.example.com"


def test_local_milvus_path_is_required_without_remote_uri(monkeypatch):
    monkeypatch.delenv("MILVUS_URI", raising=False)
    monkeypatch.setenv("MILVUS_DB_PATH", "data/test.db")

    assert get_milvus_uri() == "data/test.db"


@pytest.mark.parametrize("value", ["", "not-a-number", "1.5"])
def test_integer_config_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("TEST_INTEGER", value)

    with pytest.raises(ValueError, match="TEST_INTEGER"):
        get_optional_int("TEST_INTEGER", 2)


def test_float_config_enforces_range(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "2.5")

    with pytest.raises(ValueError, match="小于等于 2.0"):
        get_optional_float("TEST_FLOAT", 0.7, minimum=0.0, maximum=2.0)
