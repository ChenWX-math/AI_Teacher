# Phase 3 RAG 实施记录

本文记录 AI 智能教师 Phase 3 的实际实现和验证过程。目标是：用户提出问题后，系统先从教材资料中找到相关片段，再把片段交给大语言模型生成回答。

## 1. 最终架构

```text
data/<科目>/*.txt
        ↓
Document 加载
        ↓
RecursiveCharacterTextSplitter 分块
        ↓
DashScope text-embedding-v3 向量化
        ↓
Milvus Lite 本地数据库
        ↓
similarity_search + subject 过滤
        ↓
教材上下文注入 Prompt
        ↓
DeepSeek 生成回答
```

DashScope 只负责 Embedding（把文字变成向量），DeepSeek 继续负责聊天回答。两者职责不同，不能用聊天模型代替 Embedding 模型。

## 2. 环境变量

在项目目录 `AI_Teacher/.env` 中填写：

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
DASHSCOPE_API_KEY=你的DashScope密钥
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v3

# Windows 建议使用纯英文绝对路径，避免 Milvus Lite/FAISS 索引写入中文路径失败
MILVUS_DB_PATH=D:\AI_Teacher_Milvus\ai_teacher_knowledge.db
MILVUS_COLLECTION=ai_teacher_knowledge
```

程序通过 `python-dotenv` 在 `core/config.py` 中显式加载 `.env`。源代码不保存密钥，`.env` 已加入 `.gitignore`。

如果使用远程 Milvus，则改用：

```env
MILVUS_URI=https://你的Milvus地址
MILVUS_TOKEN=你的MilvusToken
```

本地 Milvus Lite 不需要 URI、用户名或 Token，只需要一个可写的 `.db` 路径。

## 3. 资料组织

资料按科目放在 `data` 下，例如：

```text
data/
├─ 数学/
│  ├─ 一次函数与二次函数.txt
│  └─ 导数及其应用.txt
├─ 物理/
├─ 化学/
└─ ...
```

`KnowledgeBase.discover_files()` 会递归读取 `.txt` 和 `.md` 文件。文件父目录名会保存为 `subject`，相对路径保存为 `source`，因此回答可以显示资料来源。

## 4. 文本切分

整本教材不能作为一个向量。`RecursiveCharacterTextSplitter` 默认使用：

- `chunk_size=800`：每个片段约 800 个字符；
- `chunk_overlap=120`：相邻片段重叠 120 个字符，避免知识点被切断；
- 中文优先按段落、换行和标点切分。

每个片段都会附带 `chunk_index`，用于定位它在原文件中的顺序。

## 5. 构建 Milvus 知识库

在项目目录执行：

```powershell
conda activate langchain1.2
cd D:\桌面\Studing\data-learning\AI_application\AI_Teacher
python -m core.rag --build
```

该命令会扫描全部 9 个科目，调用 DashScope 生成向量，并重建 `ai_teacher_knowledge` 集合。当前实测生成 74 个文本块。

只构建某个科目可以执行：

```powershell
python -m core.rag --build --subject 数学
```

统一集合的正式初始化建议使用全量 `--build`，因为它会清理旧集合，避免重复插入。单科目增量构建适合后续实现分区或删除表达式后再使用。

## 6. 独立检索测试

```powershell
python -m core.rag --query "二次函数的顶点和对称轴" --subject 数学
python -m core.rag --query "牛顿第二定律如何使用" --subject 物理
```

`KnowledgeBase.search()` 的工作顺序是：

1. 检查知识库是否已经 `build()` 或 `load()`；
2. 将问题调用同一个 DashScope Embedding 模型向量化；
3. 在 Milvus 中执行相似度搜索；
4. 通过 `subject == "数学"` 过滤当前科目；
5. 返回 `Document` 列表；
6. `format_context()` 将片段和来源整理成 Prompt 文本。

## 7. Streamlit 中的使用

主程序已经加入“启用教材知识库”复选框和 `top_k` 滑块。开启后：

```python
retrieved = knowledge_base.search(user_prompt, subject=subject, k=top_k)
context = knowledge_base.format_context(retrieved)
chain_input = {"input": user_prompt, "context": context}
```

知识库通过 `st.cache_resource` 缓存，只在 Streamlit 进程中首次使用时加载，不会每次页面重跑都重新向量化。关闭复选框时仍然使用普通对话流程。

启动应用：

```powershell
streamlit run ai_teacher_app.py
```

## 8. 已解决的问题

### `should create connection first`

`langchain-milvus 0.3.x` 同时使用新版 `MilvusClient` 和旧版 ORM `Collection`。新版客户端创建的是动态 `cm-*` 别名，旧 ORM 不会自动认识它。`core/rag.py` 中的 `CompatibleMilvus` 在初始化时为同一别名补充 `pymilvus.connections.connect()`，因此建表和索引可以正常工作。

### Windows 中文路径索引失败

Milvus Lite 底层 FAISS 在写索引文件时对中文绝对路径兼容性不稳定，可能出现：

```text
faiss::FileIOWriter ... could not open ... autoindex.idx
```

因此 `.env` 使用 `D:\AI_Teacher_Milvus\...` 这样的纯英文路径。教材资料仍然可以保留在项目的中文科目目录中。

### API 从哪里读取

`core/config.py` 使用 `load_dotenv()` 加载 `.env`，`core/llm.py` 和 `core/embeddings.py` 分别读取 DeepSeek、DashScope 密钥。不要在代码中硬编码密钥，也不要把真实 `.env` 内容发到聊天或提交到 Git。

## 9. 下一步验证清单

- 在 Streamlit 中分别关闭、开启知识库，确认普通对话和 RAG 对话都能工作；
- 测试数学、物理、化学等多个科目，确认科目过滤不会串库；
- 检查回答末尾是否显示资料来源；
- 修改一个资料文件后重新执行全量 `python -m core.rag --build`；
- 确认会话管理中的 `session_id` 配置仍然传给 `chain.stream()`。
