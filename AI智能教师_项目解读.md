# AI 智能教师（LangChain）项目解读

> 本文档记录本次任务的完整过程，并逐文件、逐技术点解读「AI 智能教师」项目。
> 配套的计划文档见同目录 `AI智能教师_LangChain项目计划.md`。

---

## 目录

1. [项目一句话说明](#1-项目一句话说明)
2. [这次任务做了什么](#2-这次任务做了什么)
3. [项目目录结构](#3-项目目录结构)
4. [逐文件详解](#4-逐文件详解)
5. [核心技术点解读](#5-核心技术点解读)
6. [关键发现与踩坑记录](#6-关键发现与踩坑记录)
7. [如何运行](#7-如何运行)
8. [下一步](#8-下一步)

---

## 1. 项目一句话说明

把一个「AI 智能伴侣」（`05.ai_application.py`，用 OpenAI SDK 直连 DeepSeek 的聊天应用）**升级重构**为「高中 9 科 AI 教师」，改用 **LangChain 架构**统一组织：

- **模型调用** → LangChain 的 `ChatOpenAI`
- **提示词** → `PromptManager` + JSON 提示词库
- **记忆（会话）** → LangChain 的 `RunnableWithMessageHistory` + 自定义 JSON 历史存储

用户只需在侧边栏选「科目 + 性别 + 性格」三个变量，就能得到一个有对应人设和教学风格的 AI 老师，且会话记忆持久化到本地。

---

## 2. 这次任务做了什么

### 2.1 Phase 0：环境准备与核实

| 事项 | 结论 |
|---|---|
| 定位项目环境 | 找到专属 conda 环境 `langchain1.2`（`D:\Program Files\miniconda3\envs\langchain1.2`，Python 3.12.13），里面已装好 langchain 1.2.12、langchain-core 1.2.18、langchain-openai 1.1.11 等全部 LangChain 包 |
| 缺失依赖 | **不是**计划里猜的 `langchain-openai`（它其实已装好），真正缺的是 **`streamlit`**，已用 `pip install streamlit` 装进 `langchain1.2`（1.63.0） |
| 模型名核实 | 计划里写的 `deepseek-chat` 已于 **2026-07-24 停用**；当前正确模型名是 **`deepseek-v4-flash`**（旧代码里写的正是这个） |
| API key | `DEEPSEEK_API_KEY` 环境变量已设置（长度 35，符合 `sk-` + 32 位格式） |

### 2.2 Phase 1：LangChain 骨架搭建

新建了以下文件（全程未改动 `05.ai_application.py` 和 `prompts/` 下已有文件）：

- `core/llm.py` —— 封装 DeepSeek 模型
- `core/prompt_builder.py` —— 提示词库 → `ChatPromptTemplate`
- `core/memory.py` —— LangChain 记忆（会话持久化）
- `core/rag.py` —— RAG 占位（Phase 3 再填）
- `core/__init__.py` —— 包标识
- `data/.gitkeep` —— RAG 知识库目录占位
- `ai_teacher_app.py` —— 主入口（Streamlit UI，含流式输出）

同时把计划里的 Step 1.6（流式打字机）一并做了。

### 2.3 目录整理

应要求，把整个项目收进 `AI_application/AI_Teacher/` 子目录，与旧的 `00~05.ai_application.py`、`A01~A05` 练习文件彻底分开。移动时把 `prompts/` 也一并移入（因为 `core` 里按「prompts 和 core 同级」来查找）。

### 2.4 排错：环境问题

运行时报 `ModuleNotFoundError: No module named 'langchain_openai'`，排查后确认是**环境没切对**：用户的 PowerShell 停在 `(base)`（miniconda 基础环境），而不是 `langchain1.2`。解决方式：

```powershell
conda activate langchain1.2
```

并把 `.vscode/settings.json` 里加了 `python.defaultInterpreterPath`，让 VS Code 默认用 `langchain1.2`。

---

## 3. 项目目录结构

```
AI_Teacher/
├── ai_teacher_app.py                  # 主入口：Streamlit UI
├── AI智能教师_LangChain项目计划.md      # 计划文档
├── AI智能教师_项目解读.md               # 本文档
├── core/                              # 核心逻辑包
│   ├── __init__.py                    # 包标识 + 模块说明
│   ├── llm.py                         # 封装 DeepSeek 模型
│   ├── prompt_builder.py              # 提示词库 → ChatPromptTemplate
│   ├── memory.py                      # LangChain 记忆（会话持久化）
│   └── rag.py                         # RAG 检索（占位）
├── prompts/                           # 提示词库（已有，未改动）
│   ├── prompt_manager.py              # PromptManager 类
│   └── subject_prompts.json           # 9 科提示词文案
├── data/                              # RAG 知识库文件目录（空，含 .gitkeep）
│   └── .gitkeep
└── sessions/                          # 会话 JSON 持久化目录（运行后自动生成）
```

---

## 4. 逐文件详解

### 4.1 `ai_teacher_app.py`（主入口）

**作用**：把「提示词 + 模型 + 记忆」串成一个 Streamlit 聊天界面。

**关键代码流程**：

```python
# 1) 页面配置：标题、图标、宽屏
st.set_page_config(page_title="AI智能教师", page_icon="🧑‍🏫", layout="wide")

# 2) 侧边栏：三个变量
subject     = st.selectbox("科目", pm.list_subjects())   # 9 科下拉
gender      = st.selectbox("性别", ["男", "女"])
personality = st.text_input("性格", placeholder="如：温柔耐心、擅长用生活例子讲解")

# 3) 组装链：提示词 | 模型，再包一层带记忆的 Runnable
def build_chain(subject, gender, personality):
    prompt = build_chat_prompt(subject, gender, personality)
    llm = get_llm()
    chain = prompt | llm
    return RunnableWithMessageHistory(
        chain, get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

# 4) 渲染历史：从 LangChain 记忆读取（单一数据源）
for msg in get_session_history(session_id).messages:
    role = "user" if msg.type == "human" else "assistant"
    st.chat_message(role).write(msg.content)

# 5) 聊天输入 + 流式输出
prompt = st.chat_input("请输入你的问题")
if prompt:
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in chain_with_history.stream({"input": prompt}, config=config):
            if chunk.content:
                full_response += chunk.content
                placeholder.markdown(full_response)
```

**几个要点**：

- `config = {"configurable": {"session_id": ...}}` 是 `RunnableWithMessageHistory` 识别「当前是哪个会话」的唯一依据。
- 历史记录**不从 `st.session_state` 存**，而是直接读磁盘（`get_session_history(...).messages`），单一数据源，避免两边不一致。
- 流式用 `st.empty()` + 累加 `chunk.content` 逐字刷新，实现「打字机」效果。
- `chunk.content` 可能为 `None`（思考阶段），用 `if chunk.content` 跳过空块。

### 4.2 `core/llm.py`（封装模型）

**作用**：统一 DeepSeek 模型入口，以后换模型 / 调参数只改这里。

```python
MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"

def get_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=BASE_URL,
        temperature=0.7,
        streaming=True,
        extra_body={"thinking": {"type": "disabled"}},
    )
```

**要点**：

- DeepSeek 是 **OpenAI 兼容协议**，所以直接用 `langchain_openai` 的 `ChatOpenAI` 即可，只需把 `base_url` 指向 DeepSeek。
- `api_key` 用 `os.environ.get("DEEPSEEK_API_KEY")` 读取，**不硬编码**密钥。
- `extra_body={"thinking": {"type": "disabled"}}` **关闭推理思考**：`deepseek-v4-flash` 默认带 thinking，关掉后走普通对话，流式输出不会出现 `delta.content` 为 `None` 的思考阶段，解析更简单。
- `model` / `api_key` / `base_url` 是 LangChain 提供的**别名**，实际字段是 `model_name` / `openai_api_key` / `openai_api_base`，两种写法都行。

### 4.3 `core/prompt_builder.py`（提示词包装）

**作用**：把 `PromptManager` 生成的**纯文本**系统提示词，包装成 LangChain 的 `ChatPromptTemplate`。

```python
def build_chat_prompt(subject, gender, personality):
    pm = PromptManager()
    system_text = pm.build_system_prompt(subject, gender, personality)

    return ChatPromptTemplate.from_messages([
        ("system", system_text),                       # 人设 + 教学守则
        MessagesPlaceholder(variable_name="history"),  # 历史占位符
        ("human", "{input}"),                          # 用户当前输入
    ])
```

**要点**：

- 模板结构**三段式**：`system`（固定人设）→ `history`（历史，由记忆组件自动填）→ `human`（当前输入）。
- `MessagesPlaceholder("history")` 是 LangChain 记忆的关键：它声明「这里会插入一个消息列表」，由 `RunnableWithMessageHistory` 在运行时把历史填进来。
- 文件顶部有一段 `sys.path` 引导，确保能 import 到 `prompts` 包（因为 `prompts/` 目录没有 `__init__.py`，靠命名空间包导入）。

### 4.4 `core/memory.py`（记忆/会话持久化）

**作用**：用 LangChain 的 `BaseChatMessageHistory` 实现会话记忆，持久化到 `sessions/{session_id}.json`。

```python
SESSIONS_DIR = os.path.join(项目根目录, "sessions")

class JsonChatHistory(BaseChatMessageHistory):
    def __init__(self, session_id, directory=None):
        self._path = os.path.join(self.directory, f"{session_id}.json")

    @property
    def messages(self):          # 读：文件不存在返回 []
        ...                      # messages_from_dict(json.load(f))

    def add_messages(self, messages):  # 批量写：读旧 + 追加新，整体落盘
        ...
        json.dump(messages_to_dict(all_messages), f, ensure_ascii=False, indent=2)

    def clear(self):             # 清空：写入 []
        ...

def get_session_history(session_id):   # 工厂函数
    return JsonChatHistory(session_id)
```

**要点**：

- **LangChain 1.x 的 `BaseChatMessageHistory` 接口变了**（详见第 5 节），只需实现 `messages`（读）、`add_messages`（批量写）、`clear`（清空）。
- 序列化用 `messages_to_dict` / `messages_from_dict`（`langchain_core.messages`），是 LangChain 官方的消息↔字典互转。
- `ensure_ascii=False`：中文**不转义**成 `\uXXXX`，文件里直接是可读中文。
- `add_messages` 采用「读旧 + 追加新 + 整体写回」的简单策略，避免覆盖丢失。

### 4.5 `core/rag.py`（RAG 占位）

**作用**：占位文件，Phase 3 再实现。计划流程：加载 `data/` 下教材 → 切片 → 向量化 → 入库 → 检索 → 注入提示词。

### 4.6 `core/__init__.py`（包标识）

**作用**：让 `core/` 成为一个可 import 的 Python 包，内容只是模块说明注释。

### 4.7 `prompts/prompt_manager.py`（已有，未改动）

**作用**：提示词库管理器，负责读 JSON 并按「科目 + 性别 + 性格」拼装系统提示词。

- `__init__`：加载 `subject_prompts.json`，拆出 `_shared`（通用）和 `_subjects`（各科专属）。
- `list_subjects()`：返回 9 个科目名。
- `build_system_prompt(subject, gender, personality)`：按**七块结构**拼出系统提示词。

### 4.8 `prompts/subject_prompts.json`（已有，未改动）

**作用**：提示词的数据文件（数据与逻辑分离）。

```json
{
  "_shared": {
    "persona_template": "你是学生的{subject}老师（性别：{gender}，教学性格：{personality}）...",
    "general_rules": [ ...7 条通用教学守则... ],
    "output_style": [ ...4 条输出风格... ]
  },
  "subjects": {
    "数学": {
      "focus": "数学培养逻辑推理...",
      "teaching_points": [ ... ],
      "question_types": [ ... ],
      "method_guide": [ ... ]
    }
    // 语文、英语、物理、化学、生物、政治、历史、地理 同结构
  }
}
```

**七块提示词结构**：人设 → 通用教学守则（7 条）→ 学科定位 → 教学侧重点 → 常见题型 → 本学科讲解方法 → 输出风格。

---

## 5. 核心技术点解读

### 5.1 LangChain 链：`prompt | llm`

`|` 是 LangChain 的 **LCEL（LangChain Expression Language）** 管道运算符：

```python
chain = prompt | llm
```

意思是「先把输入交给 prompt 格式化成消息，再把消息交给 llm 生成回复」。`chain` 本身也是一个可调用对象，可 `.invoke()`（一次性）或 `.stream()`（流式）。

### 5.2 记忆：`RunnableWithMessageHistory` 的数据流

```
用户输入 {"input": "..."}
        │
        ▼
RunnableWithMessageHistory(chain, get_session_history)
        │  ① 根据 config 里的 session_id 调 get_session_history 拿到历史
        │  ② 把历史消息填进 prompt 的 {history} 占位符
        ▼
ChatPromptTemplate（system + history + human）
        │
        ▼
ChatOpenAI（DeepSeek）──► 回复
        │
        ▼
        │  ③ 把用户输入 + AI 回复 追加写回历史（自动持久化）
```

**关键**：历史「存哪、怎么存」由 `get_session_history` 决定（这里是 `JsonChatHistory` → 本地 JSON）；「什么时候读、什么时候写」由 `RunnableWithMessageHistory` 自动完成，业务代码不用手动管。

### 5.3 LangChain 1.x 的 `BaseChatMessageHistory` 接口变化（重要）

| | 旧版（0.x） | 新版（1.x，本项目用的 1.2.18） |
|---|---|---|
| 读消息 | `messages` 抽象属性 | `messages`（类注解，需实现 property） |
| 写消息 | `add_message`（抽象） | `add_messages`（批量，推荐实现）；`add_message` 会自动委托给它 |
| 清空 | `clear`（抽象） | `clear`（仍是唯一抽象方法） |

**踩坑点**：新版里 `add_message` 会检测子类是否实现了 `add_messages`，实现了就调 `add_messages([msg])`，否则抛 `NotImplementedError`。所以本项目在 `JsonChatHistory` 里实现的是 **`add_messages`**（批量），而不是旧教程里常见的 `add_message`。

### 5.4 消息序列化：`messages_to_dict` / `messages_from_dict`

```python
# 存：BaseMessage 列表 → dict 列表 → JSON 文件
json.dump(messages_to_dict(messages), f, ensure_ascii=False)

# 读：JSON 文件 → dict 列表 → BaseMessage 列表
messages = messages_from_dict(json.load(f))
```

这两个函数在 `langchain_core.messages` 里，负责消息与字典的互转。`ensure_ascii=False` 保证中文落盘后是明文。

### 5.5 关闭 thinking：`extra_body`

`deepseek-v4-flash` 是带推理的模型，默认流式返回时会先输出一段 `reasoning_content`（思考过程），此时 `delta.content` 为 `None`。本项目通过：

```python
extra_body={"thinking": {"type": "disabled"}}
```

直接关闭思考，让模型只输出正文，流式解析就不会因 `content` 为 `None` 而丢字或报错（这对应计划里 Step 2.3 的兼容问题，已在骨架阶段提前解决）。

### 5.6 三个变量如何变成系统提示词

`subject` + `gender` + `personality` → `PromptManager.build_system_prompt()` → 用 `persona_template` 的 `.format()` 把三个变量填进人设模板，再拼上守则/学科内容/输出风格 → 一个完整的中文系统提示词字符串。

---

## 6. 关键发现与踩坑记录

1. **`langchain-openai` 其实不缺**，计划里 Step 0.1 的假设有误；真正缺的是 `streamlit`。
2. **模型名不是 `deepseek-chat`**，那个已停用（2026-07-24）；当前是 `deepseek-v4-flash`。
3. **`BaseChatMessageHistory` 接口在 1.x 有变化**：`add_message` → `add_messages`（批量），`messages` 从抽象属性变成需自行实现的 property。
4. **`ChatOpenAI` 的字段名**：`model`/`api_key`/`base_url` 是别名，底层是 `model_name`/`openai_api_key`/`openai_api_base`。
5. **流式后的消息类型是 `AIMessageChunk`**：其 `.type` 是 `"AIMessageChunk"`（不是 `"ai"`），但序列化往返内容不丢；渲染时按「human→user，其余→assistant」处理即可。
6. **环境切换是本项目最容易出错的一步**：有 base / Python313 / langchain1.2 等多个 Python，务必 `conda activate langchain1.2` 后再跑。

---

## 7. 如何运行

```powershell
conda activate langchain1.2
cd D:\桌面\Studing\data-learning\AI_application\AI_Teacher
streamlit run ai_teacher_app.py
```

**验证是否在正确环境**：

```powershell
python -c "import sys; print(sys.executable)"
# 应输出：D:\Program Files\miniconda3\envs\langchain1.2\python.exe
```

---

## 8. 下一步

- **Phase 2（体验打磨）**：异常处理健壮性、会话管理（新建/加载/删除）、thinking 字段兼容（骨架阶段已提前处理）。
- **Phase 3（RAG）**：把教材/讲义放进 `data/`，切片 → 向量化 → 入库 → 检索注入提示词，让老师能「照着教材讲」。
