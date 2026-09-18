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

from core.config import get_optional, get_optional_float, get_optional_int, get_required

# DeepSeek 当前推荐的 Flash API 名称。允许通过环境变量切换，避免模型升级时改代码。
DEFAULT_MODEL_NAME = "deepseek-flash"

# DeepSeek 的 OpenAI 兼容接口地址
DEFAULT_BASE_URL = "https://api.deepseek.com"


def get_llm(*, streaming: bool = True, temperature: float | None = None) -> ChatOpenAI:
    """返回一个封装好 DeepSeek 的 ChatOpenAI 实例。

    Returns:
        ChatOpenAI: 可直接用于 `prompt | llm` 链的模型对象。
    """
    api_key = get_required("DEEPSEEK_API_KEY")
    return ChatOpenAI(
        model=get_optional("DEEPSEEK_MODEL", DEFAULT_MODEL_NAME),
        api_key=api_key,  # type: ignore
        base_url=get_optional("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        temperature=(
            get_optional_float("DEEPSEEK_TEMPERATURE", 0.7, minimum=0.0, maximum=2.0)
            if temperature is None
            else temperature
        ),
        timeout=get_optional_float("DEEPSEEK_TIMEOUT_SECONDS", 60.0, minimum=1.0),
        max_retries=get_optional_int("DEEPSEEK_MAX_RETRIES", 2, minimum=0),
        streaming=streaming,
        extra_body={"thinking": {"type": "disabled"}},  # 关闭推理思考，直接输出正文
    )
