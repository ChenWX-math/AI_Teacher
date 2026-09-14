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


def get_milvus_uri() -> str:
    """返回远程 URI 或本地数据库路径。"""
    return get_optional("MILVUS_URI", get_required("MILVUS_DB_PATH"))
