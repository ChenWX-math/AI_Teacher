"""
LLM 模型封装模块

统一 DeepSeek 模型的入口，用 langchain-openai 的 ChatOpenAI 封装 DeepSeek
（DeepSeek 是 OpenAI 兼容协议）。以后要换模型 / 调参数，只改这一个文件即可。

说明：
    - base_url 指向 DeepSeek 官方接口
    - api_key 从环境变量 DEEPSEEK_API_KEY 读取，不硬编码
    - extra_body 关闭 thinking（deepseek-v4-flash 默认带推理，关掉后走普通对话，
      流式输出时不会出现 delta.content 为 None 的思考阶段，解析更简单）
"""

from langchain_openai import ChatOpenAI
from core.config import get_required

# DeepSeek 当前公开模型名。
# 注意：旧的 deepseek-chat / deepseek-reasoner 已于 2026-07-24 停用，
# 现用 V4 系列命名（deepseek-v4-flash / deepseek-v4-pro）。
MODEL_NAME = "deepseek-v4-flash"

# DeepSeek 的 OpenAI 兼容接口地址
BASE_URL = "https://api.deepseek.com"


def get_llm() -> ChatOpenAI:
    """返回一个封装好 DeepSeek 的 ChatOpenAI 实例。

    Returns:
        ChatOpenAI: 可直接用于 `prompt | llm` 链的模型对象。
    """
    api_key = get_required("DEEPSEEK_API_KEY")
    return ChatOpenAI(
        model=MODEL_NAME,                               # DeepSeek 模型名
        api_key=api_key,                         # type: ignore
        base_url=BASE_URL,                              # OpenAI 兼容接口地址
        temperature=0.7,                                # 生成温度：越高越发散，0.7 兼顾自然与稳定
        streaming=True,                                 # 开启流式输出，配合 .stream() 做打字机效果
        extra_body={"thinking": {"type": "disabled"}},  # 关闭推理思考，直接输出正文
    )
