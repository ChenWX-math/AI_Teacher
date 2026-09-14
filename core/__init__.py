"""AI 智能教师 —— 核心逻辑包。

包含：
    - llm.py            封装 DeepSeek 模型
    - prompt_builder.py 把提示词库包装成 ChatPromptTemplate
    - memory.py         LangChain 记忆（会话持久化）
    - session_manager.py 会话创建、切换、删除和清空
    - rag.py            教材加载、文本切分与知识检索
"""
