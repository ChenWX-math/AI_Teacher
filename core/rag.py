"""AI 教师 RAG 知识库。

流程：资料文件 -> LangChain Document -> 文本切分 -> DashScope Embedding
-> Milvus Lite/Server -> 相似度检索。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import connections

from core.config import PROJECT_ROOT, get_milvus_uri, get_optional, get_optional_float
from core.embeddings import get_embeddings

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_COLLECTION = "ai_teacher_knowledge"
DEFAULT_INDEX_PARAMS = {
    "metric_type": "L2",
    "index_type": "AUTOINDEX",
    "params": {},
}


def resolve_milvus_uri(uri: str) -> str:
    """解析本地 .db 路径；远程 http(s) 地址保持不变。"""
    if "://" in uri:
        return uri
    if Path(uri).is_absolute():
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    path = PROJECT_ROOT / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def milvus_connection_args() -> dict[str, str]:
    uri = resolve_milvus_uri(get_milvus_uri())
    args = {"uri": uri}
    if uri.startswith(("http://", "https://")):
        token = get_optional("MILVUS_TOKEN", "")
        if token:
            args["token"] = token
    return args


def ensure_milvus_connection() -> None:
    """建立 langchain-milvus 所需的 pymilvus ORM 默认连接。"""
    args = milvus_connection_args()
    if not connections.has_connection("default"):
        connections.connect(alias="default", **args) # type: ignore


class CompatibleMilvus(Milvus):
    """兼容 langchain-milvus 0.3.x 与 pymilvus 2.6 的连接别名。"""

    def _init(self, *args, **kwargs):
        # MilvusClient 在 2.6 中使用动态 ``cm-*`` 别名；ORM Collection
        # 仍需要同一别名已通过 connections.connect 注册。
        if not connections.has_connection(self.alias):
            connections.connect(alias=self.alias, **self._connection_args)
        return super()._init(*args, **kwargs)


class KnowledgeBase:
    """所有科目的教材知识库。"""

    def __init__(
        self,
        data_dir=DEFAULT_DATA_DIR,
        chunk_size=800,
        chunk_overlap=120,
        score_threshold: float | None = None,
    ):
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_size 必须大于 0，chunk_overlap 必须小于 chunk_size")
        if score_threshold is None:
            score_threshold = get_optional_float(
                "RAG_SCORE_THRESHOLD", 0.0, minimum=0.0, maximum=1.0
            )
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold 必须在 0 到 1 之间")
        self.data_dir = Path(data_dir)
        self.score_threshold = score_threshold
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        self.collection_name = get_optional("MILVUS_COLLECTION", DEFAULT_COLLECTION)
        self.vector_store: Milvus | None = None

    def discover_files(self, subject: str | None = None) -> list[Path]:
        root = self.data_dir / subject if subject else self.data_dir
        if not root.exists():
            raise FileNotFoundError(f"知识库目录不存在：{root}")
        return sorted(path for path in root.rglob("*")
                      if path.is_file() and path.suffix.lower() in {".txt", ".md"})

    def load_documents(self, subject: str | None = None) -> list[Document]:
        documents = []
        for path in self.discover_files(subject):
            text = path.read_text(encoding="utf-8-sig").strip()
            if text:
                documents.append(Document(page_content=text, metadata={
                    "source": path.relative_to(self.data_dir).as_posix(),
                    "file_name": path.name, "subject": path.parent.name,
                }))
        if not documents:
            raise ValueError(f"{subject or 'data'} 中没有可读取的 .txt 或 .md 文件")
        return documents

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        chunks = self.splitter.split_documents(list(documents))
        counters: dict[str, int] = {}
        for chunk in chunks:
            source = str(chunk.metadata.get("source", "unknown"))
            chunk.metadata["chunk_index"] = counters.get(source, 0)
            counters[source] = chunk.metadata["chunk_index"] + 1
        if not chunks:
            raise ValueError("文本切分后没有有效文本块")
        return chunks

    def build(self, subject: str | None = None, drop_old: bool = True) -> "KnowledgeBase":
        """把资料切分、向量化并写入 Milvus。"""
        ensure_milvus_connection()
        documents = self.split_documents(self.load_documents(subject))
        if subject and not drop_old:
            # 单学科增量重建时先删除该学科的旧片段，否则 auto_id 会让每次重建
            # 都追加一份重复数据。其他学科的数据保持不变。
            self.load()
            assert self.vector_store is not None
            self.vector_store.delete(expr=self._subject_filter(subject))
            self.vector_store.add_documents(documents)
            return self
        self.vector_store = CompatibleMilvus.from_documents(
            documents=documents, embedding=get_embeddings(),
            collection_name=self.collection_name,
            connection_args=milvus_connection_args(),
            index_params=DEFAULT_INDEX_PARAMS,
            drop_old=drop_old, auto_id=True,
        )
        return self

    def load(self) -> "KnowledgeBase":
        """连接已有集合，不重新向量化资料。"""
        ensure_milvus_connection()
        self.vector_store = CompatibleMilvus(
            embedding_function=get_embeddings(),
            collection_name=self.collection_name,
            connection_args=milvus_connection_args(),
            index_params=DEFAULT_INDEX_PARAMS,
        )
        return self

    def search(self, query: str, subject: str | None = None, k: int = 4) -> list[Document]:
        """兼容原有调用方式，只返回文档。"""
        return [document for document, _score in self.search_with_scores(query, subject, k)]

    def search_with_scores(
        self,
        query: str,
        subject: str | None = None,
        k: int = 4,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        """返回文档及归一化相关度分数，分数范围为 0～1，越大越相关。"""
        if self.vector_store is None:
            raise RuntimeError("知识库尚未加载，请先调用 build() 或 load()")
        if not query or not query.strip():
            raise ValueError("query 不能为空")
        if k <= 0:
            raise ValueError("k 必须大于 0")
        threshold = self.score_threshold if score_threshold is None else score_threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("score_threshold 必须在 0 到 1 之间")
        expr = self._subject_filter(subject) if subject else None
        return self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=k,
            expr=expr,
            score_threshold=threshold,
        )

    @staticmethod
    def _subject_filter(subject: str) -> str:
        """生成 Milvus 学科过滤表达式。"""
        escaped = subject.replace("\\", "\\\\").replace('"', '\\"')
        return f'subject == "{escaped}"'

    @staticmethod
    def format_context(documents: list[Document]) -> str:
        return "\n\n".join(
            f"[资料 {index} | 来源：{doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for index, doc in enumerate(documents, 1)
        )

    @staticmethod
    def source_records(
        results: list[tuple[Document, float]], preview_length: int = 180
    ) -> list[dict[str, object]]:
        """把检索结果转换为适合 UI 和日志展示的来源记录。"""
        records = []
        for index, (document, score) in enumerate(results, 1):
            content = " ".join(document.page_content.split())
            preview = content[:preview_length]
            if len(content) > preview_length:
                preview += "…"
            records.append({
                "label": f"资料 {index}",
                "source": document.metadata.get("source", "未知"),
                "chunk_index": document.metadata.get("chunk_index", "未知"),
                "score": float(score),
                "preview": preview,
            })
        return records


def build_knowledge_base(data_dir=DEFAULT_DATA_DIR, subject: str | None = None):
    return KnowledgeBase(data_dir=data_dir).build(subject=subject)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace") # type: ignore
    parser = argparse.ArgumentParser(description="构建或测试 AI 教师 Milvus 知识库")
    parser.add_argument("--build", action="store_true", help="重新构建 Milvus 集合")
    parser.add_argument("--subject", help="只构建或检索某个科目")
    parser.add_argument("--query", nargs="+", help="检索问题")
    parser.add_argument("-k", type=int, default=3, help="返回结果数量")
    args = parser.parse_args()
    kb = KnowledgeBase()
    if args.build:
        kb.build(subject=args.subject, drop_old=not bool(args.subject))
        print(f"知识库构建完成：集合={kb.collection_name}，范围={args.subject or '全部科目'}")
    if args.query:
        if kb.vector_store is None:
            kb.load()
        scored_results = kb.search_with_scores(" ".join(args.query), subject=args.subject, k=args.k)
        print(f"检索到 {len(scored_results)} 条结果")
        for source in kb.source_records(scored_results, preview_length=500):
            print(
                f"\n[{source['label']} | 相关度：{source['score']:.3f} | "
                f"来源：{source['source']}]\n{source['preview']}"
            )
    if not args.build and not args.query:
        parser.error("请提供 --build 或 --query")


if __name__ == "__main__":
    main()
