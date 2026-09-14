# AI 智能教师 Phase 2 实现教程

本文记录本次改造，并按 LangChain 初学者的学习顺序解释核心知识点。

## 1. Phase 2 目标

Phase 1 的链路是：ChatPromptTemplate → ChatOpenAI → RunnableWithMessageHistory → JSON 会话。Phase 2 将它变成可稳定使用的应用：配置错误可提示、回复流式显示、会话可新建/加载/删除/清空、损坏文件不会直接崩溃、空响应不会写入历史，同时保持旧版 05.ai_application.py 不变。

## 2. 文件职责

- ai_teacher_app.py：Streamlit 页面和交互。
- core/llm.py：模型配置和 API Key 检查。
- core/prompt_builder.py：构造聊天提示词。
- core/memory.py：JSON 持久化历史及会话 CRUD。
- prompts/prompt_manager.py：从 JSON 生成教师人设。
- core/rag.py：Phase 3 再实现。

## 3. LangChain 核心知识

### 3.1 ChatPromptTemplate

system 定义教师身份，MessagesPlaceholder("history") 接收历史消息，{input} 接收本轮问题。消息模板保留用户和助手的角色，比把历史拼成一段字符串可靠。

### 3.2 LCEL 管道

chain = prompt | llm 表示把提示词输出交给模型。每个组件都可以独立测试和替换。

### 3.3 RunnableWithMessageHistory

调用时传入 session_id，LangChain 会读取对应历史、填充 history、调用模型，并自动保存本轮用户和 AI 消息。

## 4. 本次逐步实现

### Step 1：会话 ID 与创建

新增 `create_session_id()`，使用时间戳和随机后缀降低重名概率；`create_session()` 会立即创建空 JSON 文件。这样点击“新建会话”后，侧边栏列表马上能看到它，而不必等待第一条消息。

### Step 2：JSON 记忆

JsonChatHistory.messages 捕获 JSON 解码、文件读取和格式错误，并提供文件路径。新增 list_sessions() 列出会话、delete_session() 删除会话。序列化继续使用 LangChain 的 messages_to_dict / messages_from_dict。

### Step 3：API Key 检查

get_llm() 在创建 ChatOpenAI 前检查 DEEPSEEK_API_KEY。缺失时抛出明确错误，界面可以直接告诉用户如何配置。

### Step 4：流式 chunk 解析

extract_text() 兼容 AIMessageChunk、字符串、content=None 和内容块列表。只有非空文本才刷新页面，因此 thinking 阶段不会导致错误，也不会产生空消息。

### Step 5：会话 CRUD 与 Streamlit 状态

侧边栏增加新建、加载、删除和清空按钮。新建和删除使用 `on_click` 回调，同时同步 `current_session` 与下拉框的 `session_picker`，避免 Streamlit 重跑时选择项跳回旧会话。删除后立即切换到新会话，避免继续引用已删除文件。

### Step 6：异常处理

配置和数据错误显示具体原因；其他异常提示检查网络或服务状态；模型返回空文本时只显示警告，不写入空 assistant 消息。

## 5. 运行方式

```powershell
conda activate langchain1.2
cd D:\桌面\Studing\data-learning\AI_application\AI_Teacher
$env:DEEPSEEK_API_KEY="你的密钥"
streamlit run ai_teacher_app.py
```

## 6. 验收清单

1. 缺少 API Key 时页面有明确错误。
2. 填写科目、性别、性格后可发送消息。
3. 回复逐步出现，刷新后历史仍存在。
4. 新建、加载、删除、清空会话均有效。
5. 损坏 JSON 能显示文件路径和原因。
6. 空响应不会写入空消息。
7. 05.ai_application.py 没有被修改。

## 7. 与 Phase 3 的衔接

RAG 的链路是：用户问题 → Retriever → 教材片段 → Prompt → LLM。下一阶段需要选择 Embedding 模型，加载并切分 PDF/Word/Markdown，建立 Chroma 或 FAISS 向量库，再把检索结果注入提示词。
