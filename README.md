# AI 智能教师

一个面向高中生的个人 AI 教师项目。项目使用 Streamlit 提供交互界面，使用 LangChain 组织 Prompt、模型调用和会话记忆，并使用 DashScope Embedding + Milvus Lite 构建多学科教材知识库，实现基于教材的检索增强生成（RAG）。

## 项目亮点

- 角色化教学：按学科、教师性别和教学风格生成系统 Prompt。
- 多会话管理：支持创建、切换、删除会话，历史消息持久化到本地 JSON。
- 教材 RAG：支持数学、物理、化学、历史、地理、政治、生物、英语、语文等学科资料检索。
- 本地向量库：使用 Milvus Lite 保存向量和元数据，不需要部署远程 Milvus 服务。
- 流式回答：通过 LangChain chain 流式输出 DeepSeek 回复。
- 可选知识库：关闭时是普通对话，开启后按当前科目检索教材片段并注入 Prompt。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| Web 界面 | Streamlit |
| LLM | DeepSeek OpenAI-compatible API |
| LLM 编排 | LangChain 1.x / LCEL |
| Embedding | DashScope `text-embedding-v3` |
| 向量数据库 | Milvus Lite / PyMilvus |
| 会话记忆 | `RunnableWithMessageHistory` + JSON |
| 配置管理 | `python-dotenv` |

## 目录结构

```text
AI_Teacher/
├─ ai_teacher_app.py              # Streamlit 主程序
├─ core/
│  ├─ config.py                   # .env 配置加载
│  ├─ llm.py                      # DeepSeek 模型封装
│  ├─ embeddings.py               # DashScope Embedding 封装
│  ├─ prompt_builder.py           # LangChain Prompt 构建
│  ├─ memory.py                   # JSON 会话历史
│  ├─ session_manager.py          # 会话 CRUD 和界面状态
│  ├─ rag.py                      # 文档切分、向量化、检索
│  └─ milvus_probe.py             # Milvus 连接探针
├─ prompts/
│  ├─ prompt_manager.py
│  └─ subject_prompts.json
├─ data/<科目>/*.txt              # 教材资料
├─ tests/
│  ├─ test_questions.json         # RAG 测试问题集
│  └─ test_rag_retrieval.py       # 检索测试脚本
├─ Phase3_RAG实施记录.md
├─ .env.example
└─ requirements.txt
```

## 环境要求

- Python 3.11+（当前开发环境为 Python 3.13）
- 已安装 Conda 或其他 Python 虚拟环境工具
- DeepSeek API Key
- DashScope API Key
- Windows 使用本地 Milvus Lite 时，数据库路径建议使用纯英文路径

## 安装

```powershell
conda create -n langchain1.2 python=3.13
conda activate langchain1.2
cd D:\桌面\Studing\data-learning\AI_application\AI_Teacher
pip install -r requirements.txt
pip install milvus-lite
```

`milvus-lite` 是 PyMilvus 的本地运行组件。部分环境还需要安装 `faiss-cpu`，如果 Milvus Lite 提示缺少 FAISS，可执行：

```powershell
pip install faiss-cpu
```

## 配置 API 和 Milvus

复制 `.env.example` 为 `.env`，填写真实密钥：

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
DASHSCOPE_API_KEY=你的DashScope密钥
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v3

MILVUS_DB_PATH=D:\AI_Teacher_Milvus\ai_teacher_knowledge.db
MILVUS_COLLECTION=ai_teacher_knowledge
```

程序会通过 `python-dotenv` 自动读取 `AI_Teacher/.env`。`.env`、会话记录和 Milvus 数据库已加入 `.gitignore`，不要上传或公开真实密钥。

## 构建知识库

首次运行或修改教材资料后，执行全量构建：

```powershell
python -m core.rag --build
```

构建流程为：读取 `data/` 下的文本文件，使用 `RecursiveCharacterTextSplitter` 切分，调用 DashScope 生成向量，然后写入 Milvus 集合。当前教材库包含 9 个科目，共生成 74 个文本块。

## 测试检索

```powershell
python -m core.milvus_probe
python -m core.rag --query "二次函数的顶点和对称轴" --subject 数学
python -m core.rag --query "牛顿第二定律如何使用" --subject 物理
python -m tests.test_rag_retrieval
```

检索测试会检查是否返回结果、来源是否属于预期教材文件，并统计每个学科的通过情况。Embedding 查询会产生 DashScope API 调用。

## 启动应用

```powershell
streamlit run ai_teacher_app.py
```

打开浏览器后，在侧边栏选择学科、教师性别和教学风格。勾选“启用教材知识库”后，当前问题会先检索对应科目的教材，再由 DeepSeek 根据检索片段回答。

## RAG 工作原理

```text
教材 TXT
  → LangChain Document
  → 文本分块
  → DashScope Embedding
  → Milvus Lite
  → 相似度检索 + 学科过滤
  → 上下文注入 ChatPromptTemplate
  → DeepSeek 流式回答
```

`source`、`file_name`、`subject` 和 `chunk_index` 会随文本块一起保存，因此应用可以知道回答使用了哪些教材资料。

## 当前边界

- 当前项目是个人学习和简历展示用的 MVP，不包含用户登录、权限系统和远程部署。
- Milvus Lite 数据库是本地运行产物，不提交到 GitHub；克隆项目后需要重新执行 `python -m core.rag --build`。
- 当前 Agent 和联网工具不是核心依赖。项目优先保证确定性的教材检索和教学回答流程。

## 学习文档

- `AI智能教师_完整教学文档.md`：从 LangChain 基础到项目实现的完整讲解。
- `Phase2_实现教程.md`：会话管理和健壮性实现记录。
- `Phase3_RAG实施记录.md`：Embedding、Milvus 和 RAG 实现记录。

