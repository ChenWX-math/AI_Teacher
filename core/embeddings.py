"""DashScope Embedding 模型封装。"""

from langchain_community.embeddings import DashScopeEmbeddings

from core.config import get_optional, get_required


def get_embeddings() -> DashScopeEmbeddings:
    """返回用于 RAG 向量化的 DashScope Embeddings。"""
    return DashScopeEmbeddings(
        model=get_optional("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3"),
        dashscope_api_key=get_required("DASHSCOPE_API_KEY"),
    )
