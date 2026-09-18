# AI 智能教师

一个面向高中生的个人 AI 教师项目。项目使用 Streamlit 提供交互界面，使用 LangChain Agent 根据学生意图自动选择教材检索、练习生成或答案批改工具，并使用 DashScope Embedding + Milvus Lite 构建多学科教材知识库。

## 项目亮点

- 角色化教学：按学科、教师性别和教学风格生成系统 Prompt。
- 多会话管理：支持创建、切换、删除会话，历史消息持久化到本地 JSON。
- 教材 RAG：支持数学、物理、化学、历史、地理、政治、生物、英语、语文等学科资料检索。
- 本地向量库：使用 Milvus Lite 保存向量和元数据，不需要部署远程 Milvus 服务。
- 流式回答：通过 LangChain chain 流式输出 DeepSeek 回复。
- Agent 自动模式：默认自动判断直接回答、检索教材、生成练习或批改答案。
- 经典对话模式：保留原有流式对话和手动 RAG 开关，作为稳定回退路径。
- 可解释检索：回答下方展示教材来源、片段编号、内容预览和归一化相关度。
- 检索评测：覆盖 9 个学科，输出 Recall@K、MRR 和关键词覆盖率报告。
- Agent 评测：使用真实模型和合成工具测试工具选择，不向模型发送本地教材。

## 技术栈

| 模块       | 技术                                  |
| ---------- | ------------------------------------- |
| Web 界面   | Streamlit                             |
| LLM        | DeepSeek OpenAI-compatible API        |
| LLM 编排   | LangChain 1.x / LCEL                  |
| Embedding  | DashScope`text-embedding-v3`        |
| 向量数据库 | Milvus Lite / PyMilvus                |
| 会话记忆   | `RunnableWithMessageHistory` + JSON |
| 配置管理   | `python-dotenv`                     |

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
├─ .env.example
├─ requirements-dev.txt           # 测试与代码检查依赖
├─ pyproject.toml                 # pytest / Ruff 配置
└─ requirements.txt
```

## 环境要求

- Python 3.11+（当前已验证环境为 Python 3.12）
- 已安装 Conda 或其他 Python 虚拟环境工具
- DeepSeek API Key
- DashScope API Key
- Windows 使用本地 Milvus Lite 时，数据库路径建议使用纯英文路径

## 安装

```powershell
conda create -n langchain1.2 python=3.12
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
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_TIMEOUT_SECONDS=60
DEEPSEEK_MAX_RETRIES=2
DASHSCOPE_API_KEY=你的DashScope密钥
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v3
RAG_SCORE_THRESHOLD=0.0

MILVUS_DB_PATH=D:\AI_Teacher_Milvus\ai_teacher_knowledge.db
MILVUS_COLLECTION=ai_teacher_knowledge
```

程序会通过 `python-dotenv` 自动读取 `AI_Teacher/.env`。`.env`、会话记录和 Milvus 数据库已加入 `.gitignore`，不要上传或公开真实密钥。

如果使用远程 Milvus，可改为配置 `MILVUS_URI` 和可选的 `MILVUS_TOKEN`；此时不需要 `MILVUS_DB_PATH`。

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
python -m tests.evaluate_agent_routing
```

检索评测覆盖 9 个学科、42 道问题，会统计 Recall@1、Recall@3、MRR 和关键词覆盖率，并在 `evaluation_results/` 下生成 JSON 与 Markdown 报告。Embedding 查询会产生 DashScope API 调用。

当前基线结果：Recall@1 `97.6%`、Recall@3 `100.0%`、MRR `0.988`、关键词覆盖率 `94.0%`。详细结果见 [RAG 检索评测](evaluation_results/rag_evaluation.md)。

Agent 路由基线为 `9/9 (100%)`；第一次评测为 `6/9 (66.7%)`，收紧工具优先级和必须调用规则后全部通过。详细结果见 [Agent 路由评测](evaluation_results/agent_routing.md)。

不调用外部 API 的单元测试：

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

单元测试使用临时目录和纯本地逻辑，不会产生模型费用。`tests.test_rag_retrieval` 属于需要已构建向量库和 DashScope API 的集成评测，应按需单独运行。

## 启动应用

```powershell
streamlit run ai_teacher_app.py
```

打开浏览器后，在侧边栏选择学科、教师性别和教学风格。默认开启“Agent 自动模式”，用户无需手动决定是否检索。例如：

- “二次函数的顶点公式是什么？”→ `search_textbook`
- “给我出一道二次函数基础题，先不要答案。”→ `generate_exercise`
- “题目是……我的答案是……请批改。”→ `grade_answer`
- “今天学习有点累，鼓励我一下。”→ 直接回答，不调用工具

关闭 Agent 自动模式后，会回到原有经典对话，可手动选择是否启用教材知识库。

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

## Agent 工作原理

```text
学生请求
  → DeepSeek 判断意图
  → search_textbook / generate_exercise / grade_answer / 直接回答
  → 工具结果返回 Agent
  → 组织最终教师回答
  → 保存用户消息和最终回答
```

工具调用轨迹不会混入用户可见的聊天历史；界面只显示最终回答、使用过的工具名称和真实教材来源。`AGENT_RECURSION_LIMIT` 限制单次 Agent 的最大执行步数，防止异常循环。

`source`、`file_name`、`subject` 和 `chunk_index` 会随文本块一起保存，因此应用可以知道回答使用了哪些教材资料。

检索相关度统一映射到 `0～1`，数值越大代表越相关。可通过 `RAG_SCORE_THRESHOLD` 过滤低相关片段；默认 `0.0` 表示先保留结果并通过评测观察分数分布，再按实际数据选择阈值，避免凭经验误删有效资料。

## 当前边界

- 不包含用户登录、权限系统和远程部署。
- Milvus Lite 数据库是本地运行产物，不提交到 GitHub；克隆项目后需要重新执行 `python -m core.rag --build`。
- 当前练习题及参考答案尚未作为独立教学状态持久化；跨轮“刚才那道题”的可靠关联将在后续教学状态阶段完善。
- 当前评测集由项目教材人工构造，不是独立公开基准；关键词覆盖率是字面匹配指标，应与人工检查和后续回答忠实度评测结合使用。

## 开发检查

提交代码前建议运行：

```powershell
python -m compileall -q ai_teacher_app.py core prompts tests
python -m pytest -q
ruff check ai_teacher_app.py core prompts tests
```
