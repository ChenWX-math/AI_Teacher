"""项目配置加载。

密钥保存在 AI_Teacher/.env，程序启动时加载为环境变量；源代码不保存真实密钥。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

# 显式指定路径，避免从不同工作目录启动 Streamlit 时找不到 .env。
load_dotenv(ENV_FILE, override=False)

# 兼容早期把本地 .db 路径写在 MILVUS_URI 的配置；PyMilvus 会读取同名
# 环境变量并把它当作远程 URL，因此本地路径需要转换成独立变量。
_legacy_milvus_uri = os.getenv("MILVUS_URI", "").strip()
if _legacy_milvus_uri and "://" not in _legacy_milvus_uri:
    os.environ.setdefault("MILVUS_DB_PATH", _legacy_milvus_uri)
    os.environ.pop("MILVUS_URI", None)


def get_required(name: str) -> str:
    """读取必填配置；缺少时给出可操作的错误。"""
    value = os.getenv(name, "").strip()
    if not value:
        raise EnvironmentError(f"缺少配置 {name}，请在 {ENV_FILE} 中填写。")
    return value


def get_optional(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def get_optional_int(name: str, default: int, minimum: int | None = None) -> int:
    """读取可选整数配置，并在配置错误时给出明确提示。"""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"配置 {name} 必须是整数，当前值为：{raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"配置 {name} 必须大于等于 {minimum}，当前值为：{value}")
    return value


def get_optional_float(
    name: str, default: float, minimum: float | None = None, maximum: float | None = None
) -> float:
    """读取可选浮点数配置，并验证取值范围。"""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"配置 {name} 必须是数字，当前值为：{raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"配置 {name} 必须大于等于 {minimum}，当前值为：{value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"配置 {name} 必须小于等于 {maximum}，当前值为：{value}")
    return value


def get_milvus_uri() -> str:
    """返回远程 URI 或本地数据库路径。"""
    # 不把 get_required() 放进默认参数：Python 会先计算函数参数，导致即使已经
    # 配置远程 MILVUS_URI，仍错误地要求 MILVUS_DB_PATH。
    remote_uri = os.getenv("MILVUS_URI", "").strip()
    return remote_uri or get_required("MILVUS_DB_PATH")


def get_storage_backend() -> str:
    backend = get_optional("STORAGE_BACKEND", "json").lower()
    if backend not in {"json", "postgres"}:
        raise ValueError("STORAGE_BACKEND 必须是 json 或 postgres")
    return backend


def get_app_backend() -> str:
    backend = get_optional("APP_BACKEND", "local").lower()
    if backend not in {"local", "api"}:
        raise ValueError("APP_BACKEND 必须是 local 或 api")
    return backend


def get_fastapi_host() -> str:
    return get_optional("FASTAPI_HOST", "127.0.0.1")


def get_fastapi_port() -> int:
    return get_optional_int("FASTAPI_PORT", 8000, minimum=1)
