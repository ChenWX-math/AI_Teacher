"""测试 Milvus Lite 或远程 Milvus 连接。

运行：
    python -m core.milvus_probe

脚本会创建一个临时测试集合，插入两条 3 维向量，执行一次相似度搜索，
最后删除测试集合，不会触碰正式知识库集合。
"""

from __future__ import annotations

import uuid
from pathlib import Path

from pymilvus import MilvusClient

from core.config import ENV_FILE, PROJECT_ROOT, get_milvus_uri, get_optional


def resolve_uri(uri: str) -> str:
    """把相对本地 .db 路径解析到项目根目录；URL 保持原样。"""
    if "://" in uri or Path(uri).is_absolute():
        return uri
    path = PROJECT_ROOT / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def main() -> None:
    raw_uri = get_milvus_uri()
    uri = resolve_uri(raw_uri)
    token = get_optional("MILVUS_TOKEN", "") if raw_uri.startswith(("http://", "https://")) else ""
    collection = f"_ai_teacher_probe_{uuid.uuid4().hex[:10]}"

    print(f"配置文件：{ENV_FILE}")
    print(f"连接地址：{uri}")
    print(f"测试集合：{collection}")

    client = MilvusClient(uri=uri, token=token)
    try:
        client.create_collection(
            collection_name=collection,
            dimension=3,
            metric_type="COSINE",
            consistency_level="Strong",
        )
        client.insert(
            collection_name=collection,
            data=[
                {"id": 1, "vector": [1.0, 0.0, 0.0], "text": "数学测试片段"},
                {"id": 2, "vector": [0.0, 1.0, 0.0], "text": "物理测试片段"},
            ],
        )
        result = client.search(
            collection_name=collection,
            data=[[0.95, 0.05, 0.0]],
            limit=1,
            output_fields=["text"],
        )
        hit = result[0][0]
        print(f"搜索成功：id={hit['id']}，距离={hit['distance']}，内容={hit['entity']['text']}")
        print("Milvus 本地/远程连接、建表、写入和检索均正常。")
    finally:
        if client.has_collection(collection):
            client.drop_collection(collection)


if __name__ == "__main__":
    main()
