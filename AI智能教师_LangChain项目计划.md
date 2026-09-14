# AI 智能教师（LangChain）项目计划

> 把一个「AI 智能伴侣」重做升级为「AI 智能教师」，改用 LangChain 架构重新组织：
> 模型调用、提示词库、记忆（会话）全部走 LangChain，主题换成「高中 9 科 AI 教师」。
> 最后阶段接入 RAG，让老师能「照着教材讲」。

---

## 一、项目目标

把现有 `05.ai_application.py`（OpenAI SDK 直连 DeepSeek 的伴侣聊天）升级为：

1. **LangChain 调用模型** —— 用 `ChatOpenAI`（OpenAI 兼容协议）封装 DeepSeek。
2. **提示词库** —— 用一个 `PromptManager` 类 + JSON 文件管理 9 个科目提示词。
3. **三个定制变量** —— 用户只需输入「科目 + 性别 + 性格」，自动生成系统提示词。
4. **LangChain 记忆** —— 用 `RunnableWithMessageHistory` + 自定义 `BaseChatMessageHistory` 管理会话，持久化到本地 JSON。
5. **RAG（最后阶段）** —— 教材/讲义切片入库，检索后注入提示词，让老师照资料讲。

> ⚠️ 原则：**全程不修改 `05.ai_application.py`**，所有新代码放在新文件里。

---

## 二、技术栈

| 层 | 用的东西 |
|---|---|
| 界面 | Streamlit |
| 模型 | DeepSeek（OpenAI 兼容接口） |
| LangChain 封装 | `langchain-openai`（`ChatOpenAI`）+ `langchain-core`（prompt / message history） |
| 提示词 | JSON 数据 + `PromptManager` 类（**已建好**） |
| 记忆 | `RunnableWithMessageHistory` + 自定义 `JsonChatHistory` |
| RAG（后） | `langchain-text-splitters` + Embeddings + 向量库（Chroma/FAISS） |

当前 `requirements.txt` 已含 `langchain==1.2.12`、`langchain-core`、`langchain-classic`、`langchain-community`、`langchain-text-splitters`、`langgraph` 等。**大概率只缺 `langchain-openai`**（见 Step 0.1）。

---

## 三、目标目录结构

```
data-learning/AI_application/
├── 05.ai_application.py            # 旧代码，一律不动
├── AI智能教师_LangChain项目计划.md   # 本文件
├── ai_teacher_app.py               # 【新】主入口（Streamlit UI）
├── core/                           # 【新】核心逻辑
│   ├── __init__.py
│   ├── llm.py                      # 【新】封装 DeepSeek 模型
│   ├── prompt_builder.py           # 【新】把提示词库包装成 ChatPromptTemplate
│   ├── memory.py                   # 【新】LangChain 记忆（会话持久化）
│   └── rag.py                      # 【新·后】RAG 检索（最后阶段再填）
├── prompts/                        # 【已有，勿动】
│   ├── subject_prompts.json        # 9 科提示词文案
│   └── prompt_manager.py           # PromptManager 类
├── data/                           # 【新·后】放 RAG 知识库文件（教材/讲义）
└── sessions/                       # 【已有】会话 JSON 持久化目录
```

---

## 四、核心设计要点（先想清楚再动手）

### 4.1 三个变量

| 变量 | 含义 | 例子 |
|---|---|---|
| `subject` | 科目 | 数学 |
| `gender` | 性别 | 女 |
| `personality` | 性格/教学风格 | 温柔耐心、擅长用生活例子讲解 |

由 `PromptManager.build_system_prompt(subject, gender, personality)` 生成系统提示词（**已实现**）。

### 4.2 提示词七块结构

人设 → 通用教学守则（7 条）→ 学科定位 → 教学侧重点 → 常见题型 → 本学科讲解方法 → 输出风格。

### 4.3 模型调用方式

DeepSeek 是 OpenAI 兼容协议，LangChain 里用：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",              # 需核实，见 Step 0.2
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0.7,
    streaming=True,
)
```

### 4.4 记忆方式（LangChain 1.x 正统做法）

```
用户输入 "input"
      │
      ▼
RunnableWithMessageHistory(chain, get_session_history=...)
      │  内部自动把历史消息填进 {history}
      ▼
ChatPromptTemplate（system + history + human）
      │
      ▼
ChatOpenAI（DeepSeek）──> 回复
```

历史存哪、怎么存，由自定义的 `JsonChatHistory(BaseChatMessageHistory)` 决定——读写 `sessions/{session_id}.json`。

---

## 五、分阶段步骤

> 用法：**每一步给 AI 的提示词都是可以直接复制粘贴给 Claude Code 的**。建议一次只做一步，做完检查「验收标准」再进下一步。

---

### Phase 0：环境准备

#### Step 0.1 安装缺失依赖

- **目标**：装上 LangChain 调 DeepSeek 所需的包。
- **产出**：更新 `requirements.txt`（或直接 pip 安装）。
- **做什么**：
  1. 确认是否缺 `langchain-openai`。
  2. 缺则安装。

- **给 AI 的提示词**（直接复制）：

```text
帮我确认当前项目（data-learning/AI_application）是否缺少 langchain-openai 包，
用于在 LangChain 里调用 DeepSeek 的 OpenAI 兼容接口。
当前 requirements.txt 已有 langchain==1.2.12、langchain-core、langchain-classic、
langchain-community、langchain-text-splitters、langgraph 等。
请：
1) 判断是否缺 langchain-openai；
2) 缺的话给出安装命令（或帮我加到 requirements.txt）；
3) 确认整个过程不修改 05.ai_application.py。
```

- **验收标准**：能 `from langchain_openai import ChatOpenAI` 不报错。

#### Step 0.2 确认密钥与模型名

- **目标**：确认 API key 和正确的 DeepSeek 模型名。
- **做什么**：
  1. 核实 DeepSeek 官方公开模型名（`deepseek-chat` / `deepseek-reasoner`）。
  2. 确认 `DEEPSEEK_API_KEY` 环境变量已设置。

- **给 AI 的提示词**：

```text
帮我核实两件事：
1) DeepSeek 官方公开的模型名到底是什么（deepseek-chat / deepseek-reasoner？）；
   现有旧代码里写的是 "deepseek-v4-flash"，请确认这个是不是真实可用的模型名。
2) 本机 DEEPSEEK_API_KEY 环境变量是否已设置。
注意：不要修改任何文件，只做核实和说明。
```

- **验收标准**：明确 `core/llm.py` 里 `model=` 该填什么。

---

### Phase 1：LangChain 骨架（核心）

#### Step 1.1 搭建目录结构

- **目标**：建好项目骨架。
- **产出**：`core/`、`data/` 目录及占位文件。

- **给 AI 的提示词**：

```text
在 data-learning/AI_application/ 下帮我搭建「AI智能教师」项目的目录骨架，新建：
- core/__init__.py
- core/llm.py（先空文件，写一行占位注释）
- core/prompt_builder.py（先空文件，写一行占位注释）
- core/memory.py（先空文件，写一行占位注释）
- core/rag.py（先空文件，写一行占位注释）
- data/（空目录，放一个 .gitkeep）
不要动 05.ai_application.py 和 prompts/ 下已有的两个文件。
```

- **验收标准**：目录结构如「三、目标目录结构」所示。

#### Step 1.2 封装 LLM（core/llm.py）

- **目标**：统一模型入口，以后只在这里改模型配置。
- **产出**：`core/llm.py`。

- **给 AI 的提示词**：

```text
新建并完成 core/llm.py，写一个 get_llm() 函数，用 langchain_openai 的 ChatOpenAI
封装 DeepSeek：
- base_url = "https://api.deepseek.com"
- api_key 从环境变量 DEEPSEEK_API_KEY 读取
- model 用 Step 0.2 核实后的名字（deepseek-chat）
- temperature=0.7、streaming=True
- 返回 ChatOpenAI 实例
要求：中文注释写清楚每个参数；用 os.environ.get 读密钥，不要硬编码；
不修改 05.ai_application.py 和 prompts/ 下已有文件。
```

- **验收标准**：`get_llm()` 能返回一个 `ChatOpenAI` 实例，不报错。

#### Step 1.3 接入提示词库（core/prompt_builder.py）

- **目标**：把 `PromptManager` 产出的字符串 prompt，包装成 LangChain 的 `ChatPromptTemplate`。
- **产出**：`core/prompt_builder.py`。

- **给 AI 的提示词**：

```text
新建并完成 core/prompt_builder.py，写一个函数（如 build_chat_prompt(subject, gender, personality)）：
1) 复用 prompts/prompt_manager.py 里的 PromptManager：
   - pm = PromptManager()
   - system_text = pm.build_system_prompt(subject, gender, personality)
2) 用 langchain_core.prompts 包装成 ChatPromptTemplate：
   ChatPromptTemplate.from_messages([
       ("system", system_text),
       MessagesPlaceholder(variable_name="history"),
       ("human", "{input}"),
   ])
3) 返回这个 ChatPromptTemplate
要求：不修改 prompts/ 下已有文件，不修改 05.ai_application.py。
```

- **验收标准**：`build_chat_prompt("数学", "女", "耐心")` 返回一个含 `history` 占位符的 `ChatPromptTemplate`。

#### Step 1.4 记忆管理（core/memory.py）

- **目标**：用 LangChain 的 `BaseChatMessageHistory` 实现会话记忆，持久化到 `sessions/{session_id}.json`。
- **产出**：`core/memory.py`。

- **给 AI 的提示词**：

```text
新建并完成 core/memory.py，实现 LangChain 的记忆管理（LangChain 1.x 风格）：
1) 定义 JsonChatHistory(BaseChatMessageHistory) 类：
   - 构造时接收 session_id，消息持久化到 sessions/{session_id}.json
   - 序列化用 langchain_core.messages 里的 messages_to_dict / messages_from_dict
   - 实现 messages 属性（读）和 add_message（写）、clear（清空）方法
   - JSON 读写用 encoding="utf-8"、ensure_ascii=False
2) 提供 get_session_history(session_id) 工厂函数，返回对应 JsonChatHistory 实例
要求：不修改 05.ai_application.py；这是给 RunnableWithMessageHistory 用的历史存储。
```

- **验收标准**：能 `add_message` 后从 `messages` 读回，且文件里中文不转义。

#### Step 1.5 串联主应用（ai_teacher_app.py）

- **目标**：把「提示词 + 模型 + 记忆」串成一个 AI 教师聊天界面。
- **产出**：`ai_teacher_app.py`。

- **给 AI 的提示词**：

```text
新建 ai_teacher_app.py，用 Streamlit 实现「AI智能教师」：
1) st.set_page_config 标题「AI智能教师」、icon 用 🧑‍🏫、layout="wide"
2) 侧边栏三个输入：
   - 科目：下拉框，选项来自 PromptManager().list_subjects()
   - 性别：下拉框（男/女）
   - 性格：文本输入（如「温柔耐心、擅长用生活例子讲解」）
3) 用 RunnableWithMessageHistory 把链串起来：
   - prompt = build_chat_prompt(科目, 性别, 性格)   # core/prompt_builder.py
   - llm = get_llm()                                # core/llm.py
   - chain = prompt | llm
   - 用 core/memory.py 的 get_session_history 做历史
   - input_messages_key="input"、history_messages_key="history"
   - session_id 用时间戳（复用 datetime 生成）
4) 聊天输入框 st.chat_input 收发消息，渲染历史用 st.chat_message
5) 未填「科目/性别/性格」前提示用户先填写
要求：不修改 05.ai_application.py；不要用旧代码的 %s 模板和手写 json.dump。
```

- **验收标准**：跑 `streamlit run ai_teacher_app.py`，能选科目/性别/性格并正常对话。

#### Step 1.6 流式输出（打字机效果）

- **目标**：回复逐字出现，而不是整段蹦出来。
- **产出**：修改 `ai_teacher_app.py`。

- **给 AI 的提示词**：

```text
在 ai_teacher_app.py 里给 AI 回复加流式输出（打字机效果）：
用 chain_with_history.stream({"input": prompt}, config={...}) 逐 chunk 接收，
配合 st.write_stream 或 st.empty() 实时刷新。
可参考 05.ai_application.py 里 st.empty() + 累加 chunk 的写法，但接口换成 LangChain 的 stream。
不要修改 05.ai_application.py 本身。
```

- **验收标准**：回复逐字出现，不是等全部生成完才显示。

---

### Phase 2：体验打磨

#### Step 2.1 异常处理与健壮性

- **给 AI 的提示词**：

```text
给 ai_teacher_app.py 补异常处理：
1) 密钥缺失或网络请求失败时不崩溃，用 st.error 给出友好提示；
2) AI 返回为空时，不往消息历史里塞空消息；
3) 会话 JSON 损坏或加载失败时，提示具体原因（不要用空的 except 吞掉错误）；
不要修改 05.ai_application.py。
```

- **验收标准**：故意删掉 API key 再运行，界面友好报错而非闪退。

#### Step 2.2 会话管理（新建/加载/删除）

- **给 AI 的提示词**：

```text
在 ai_teacher_app.py 侧边栏加会话管理：
1) 「新建会话」按钮：切换到一个新的 session_id（时间戳），清空当前消息；
2) 会话历史列表：列出 sessions/ 目录下所有会话，可加载；
3) 删除按钮：删除对应 sessions/{session_id}.json；
要求：这些操作与 core/memory.py 的 JsonChatHistory 存储对应；
可参考 05.ai_application.py 的会话 CRUD 逻辑，但改造成配合 LangChain 记忆；
不修改 05.ai_application.py。
```

- **验收标准**：能新建、切换、删除会话，且各会话历史互不串。

#### Step 2.3 推理模型 thinking 字段兼容

- **给 AI 的提示词**：

```text
核实 DeepSeek 模型流式返回时是否带 reasoning_content（思考内容）字段；
如果有，确保 ai_teacher_app.py 的流式解析兼容——不要因为 delta.content 为 None 就漏掉真实内容。
不要修改 05.ai_application.py。
```

- **验收标准**：流式输出不丢字、不报错。

---

### Phase 3：RAG（最后阶段，概要）

> 先跑通 Phase 1–2，再做这步。让 AI 教师能「照着教材/讲义讲」，避免瞎编。

| Step | 内容 | 一句话说明 |
|---|---|---|
| 3.1 | 准备资料 | 把教材/讲义 PDF、Word、Markdown 放进 `data/` |
| 3.2 | 加载+切片 | `DocumentLoader` 读文件 → `RecursiveCharacterTextSplitter` 切片 |
| 3.3 | 向量化+入库 | Embeddings（选 Ollama 本地 / OpenAI / SiliconFlow）→ Chroma 或 FAISS |
| 3.4 | 检索器+RAG 链 | `vectorstore.as_retriever()` 检索相关知识点 → 塞进提示词 |
| 3.5 | UI 开关 | 侧边栏加「知识库开关」，开=走 RAG，关=纯对话 |

- **给 AI 的提示词（概要，做之前再细化）**：

```text
给 AI 教师接入 RAG（先出方案再动手）：
1) 决定 Embedding 用哪个（优先 Ollama 本地 nomic-embed-text/bge-m3，其次 OpenAI 或 SiliconFlow BGE）；
2) 把 data/ 下的教材加载、切片、向量化、入库（Chroma 或 FAISS）；
3) 写 core/rag.py：检索器 + 把检索到的知识点注入提示词；
4) 在 ai_teacher_app.py 加「知识库开关」。
注意：DeepSeek 没有公开 Embedding API，不能直接用 DeepSeek 做向量化。
不要修改 05.ai_application.py。
```

---

## 六、LangChain 1.x 避坑清单（务必看）

1. **导入路径**：`ChatOpenAI` 在 `langchain_openai`，不是老的 `langchain.chat_models`。
2. **记忆拆包**：`ConversationBufferMemory` 在 `langchain-classic`（旧）；推荐用 `RunnableWithMessageHistory`（在 `langchain_core.runnables.history`）。
3. **占位符**：用 `{input}`、`{history}`（`MessagesPlaceholder`），别再用 `%s`。
4. **模型名**：别照搬旧代码的 `deepseek-v4-flash`，用核实后的 `deepseek-chat`。
5. **DeepSeek 无公开 Embedding API**：RAG 的向量化要另选 Embedding 模型。
6. **流式兼容 thinking**：推理模型 `delta.content` 可能为 `None`，注意别丢内容。

---

## 七、进度跟踪

- [x] 提示词库（`prompts/subject_prompts.json` + `prompts/prompt_manager.py`）
- [ ] Phase 0：环境准备
- [ ] Phase 1：LangChain 骨架
- [ ] Phase 2：体验打磨
- [ ] Phase 3：RAG
