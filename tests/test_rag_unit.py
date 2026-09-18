"""不连接 Embedding API 或 Milvus 的 RAG 单元测试。"""

from pathlib import Path

import pytest
from langchain_core.documents import Document

import core.rag as rag_module
from core.rag import KnowledgeBase, resolve_milvus_uri


def test_discovers_and_loads_subject_documents(tmp_path):
    subject_dir = tmp_path / "数学"
    subject_dir.mkdir()
    (subject_dir / "函数.txt").write_text("二次函数的顶点与对称轴。", encoding="utf-8")
    (subject_dir / "ignore.pdf").write_text("不会读取", encoding="utf-8")

    kb = KnowledgeBase(data_dir=tmp_path, chunk_size=20, chunk_overlap=5)
    documents = kb.load_documents("数学")
    chunks = kb.split_documents(documents)

    assert len(documents) == 1
    assert documents[0].metadata["source"] == "数学/函数.txt"
    assert all(chunk.metadata["subject"] == "数学" for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_invalid_chunk_configuration_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="chunk_overlap"):
        KnowledgeBase(data_dir=tmp_path, chunk_size=100, chunk_overlap=100)


def test_subject_filter_escapes_special_characters():
    assert KnowledgeBase._subject_filter('数\\学"实验') == 'subject == "数\\\\学\\"实验"'


def test_relative_milvus_path_is_resolved_inside_project():
    resolved = Path(resolve_milvus_uri("data/milvus/test.db"))

    assert resolved.is_absolute()
    assert resolved.name == "test.db"


def test_incremental_subject_build_replaces_old_subject_chunks(monkeypatch, tmp_path):
    calls = []

    class FakeVectorStore:
        def delete(self, *, expr):
            calls.append(("delete", expr))

        def add_documents(self, documents):
            calls.append(("add", documents))

    document = Document(page_content="函数内容", metadata={"subject": "数学"})
    kb = KnowledgeBase(data_dir=tmp_path)

    monkeypatch.setattr(rag_module, "ensure_milvus_connection", lambda: None)
    monkeypatch.setattr(kb, "load_documents", lambda subject: [document])
    monkeypatch.setattr(kb, "split_documents", lambda documents: documents)

    def fake_load():
        kb.vector_store = FakeVectorStore()
        return kb

    monkeypatch.setattr(kb, "load", fake_load)

    kb.build(subject="数学", drop_old=False)

    assert calls == [
        ("delete", 'subject == "数学"'),
        ("add", [document]),
    ]


def test_search_with_scores_uses_subject_and_threshold(tmp_path):
    document = Document(
        page_content="二次函数顶点公式",
        metadata={"source": "数学/函数.txt", "chunk_index": 2},
    )

    class FakeVectorStore:
        def similarity_search_with_relevance_scores(self, query, **kwargs):
            assert query == "顶点怎么求"
            assert kwargs == {
                "k": 3,
                "expr": 'subject == "数学"',
                "score_threshold": 0.4,
            }
            return [(document, 0.92)]

    kb = KnowledgeBase(data_dir=tmp_path, score_threshold=0.4)
    kb.vector_store = FakeVectorStore()

    results = kb.search_with_scores("顶点怎么求", subject="数学", k=3)

    assert results == [(document, 0.92)]
    assert kb.search("顶点怎么求", subject="数学", k=3) == [document]


def test_source_records_are_safe_and_compact():
    document = Document(
        page_content="第一行\n第二行很长的内容",
        metadata={"source": "数学/函数.txt", "chunk_index": 1},
    )

    records = KnowledgeBase.source_records([(document, 0.876)], preview_length=8)

    assert records == [{
        "label": "资料 1",
        "source": "数学/函数.txt",
        "chunk_index": 1,
        "score": 0.876,
        "preview": "第一行 第二行很…",
    }]


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_score_threshold_is_rejected(tmp_path, threshold):
    with pytest.raises(ValueError, match="score_threshold"):
        KnowledgeBase(data_dir=tmp_path, score_threshold=threshold)
