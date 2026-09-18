"""AI 智能教师 Streamlit 应用（Phase 2）。"""
import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory

from core.context_manager import LayeredContextManager
from core.llm import get_llm
from core.memory import get_session_history
from core.prompt_builder import build_chat_prompt
from core.rag import KnowledgeBase
from core.session_manager import SessionManager
from core.teacher_agent import build_teacher_agent, run_teacher_agent
from core.teacher_tools import create_teacher_tools
from core.teaching_state import load_teaching_state, save_teaching_state
from prompts.prompt_manager import PromptManager

st.set_page_config(page_title="AI智能教师", page_icon="🧑‍🏫", layout="wide")
pm = PromptManager()
session_manager = SessionManager()

@st.cache_resource(show_spinner=False)
def get_knowledge_base():
    return KnowledgeBase().load()


def build_chain(subject, gender, personality, use_context=False):
    """构建带历史记录的对话链。

    先根据科目、性别和性格构建聊天提示模板，再与 LLM 组合成可运行链，
    最后通过 RunnableWithMessageHistory 包装，使链具备对话历史记忆能力。

    Args:
        subject: 科目名称，如 "数学"、"语文" 等
        gender: 教师性别，"男" 或 "女"
        personality: 教师性格描述，如 "温柔耐心、擅长用生活例子讲解"

    Returns:
        RunnableWithMessageHistory: 包装了消息历史的可运行链对象
    """
    chain = build_chat_prompt(subject, gender, personality, use_context=use_context) | get_llm()
    return RunnableWithMessageHistory(chain, get_session_history,
        input_messages_key="input", history_messages_key="history")

def extract_text(chunk):
    """从 LLM 流式输出的 chunk 中提取文本内容。

    兼容多种 chunk 格式：
    - 带 content 属性的对象（如 AIMessageChunk）
    - 直接传入的字符串
    - 包含多个 dict 的列表（如 OpenAI 格式的 choices）

    Args:
        chunk: LLM 流式输出的单个数据块，支持多种类型

    Returns:
        str: 提取出的文本字符串，若 chunk 为 None 则返回空字符串
    """
    content = getattr(chunk, "content", chunk)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
    return str(content)


def render_sources(source_records):
    """在回答下方展示本次检索实际使用的教材来源。"""
    if not source_records:
        return
    with st.expander(f"📚 参考教材（{len(source_records)} 条）"):
        for source in source_records:
            st.markdown(
                f"**{source['label']} · {source['source']} · "
                f"片段 {source['chunk_index']} · 相关度 {source['score']:.1%}**"
            )
            st.text(source["preview"])

with st.sidebar:
    st.subheader("AI 教师设置")
    # on_click 回调在 widget 渲染前执行，避免 StreamlitWidgetAlreadyInstantiatedError
    st.button("➕ 新建会话", use_container_width=True,
              on_click=session_manager.new_session)

    sessions = session_manager.sessions
    if sessions:
        current = session_manager.current_session
        # 若当前会话不在列表中（被删除后的短暂状态），以第一个为准
        index = sessions.index(current) if current in sessions else 0
        st.selectbox("加载会话", sessions, index=index,
                     key=session_manager.PICKER_KEY,
                     on_change=session_manager.load_selected_session)
        st.button("🗑️ 删除当前会话", use_container_width=True,
                  on_click=session_manager.delete_current_session)

    subjects = pm.list_subjects()
    # key 缺失说明刚切换到新会话（_apply_session_settings 删掉了它），补入默认值
    if session_manager.SUBJECT_KEY not in st.session_state:
        st.session_state[session_manager.SUBJECT_KEY] = subjects[0]
    subject = st.selectbox("科目", subjects, key=session_manager.SUBJECT_KEY)
    gender = st.selectbox("性别", ["男", "女"], key=session_manager.GENDER_KEY)
    personality = st.text_input("性格", placeholder="如：温柔耐心、擅长用生活例子讲解",
                                key=session_manager.PERSONALITY_KEY)
    agent_mode = st.toggle(
        "Agent 自动模式",
        value=True,
        help="开启后由 AI 自动判断直接回答、检索教材、生成练习或批改答案。",
    )
    use_knowledge = False
    if not agent_mode:
        use_knowledge = st.checkbox(
            "启用教材知识库",
            value=False,
            help="经典模式下手动决定是否先检索当前科目教材。",
        )
    top_k = st.slider("检索片段数", min_value=1, max_value=6, value=3,
                      disabled=not (agent_mode or use_knowledge))
    if st.button("🧹 清空当前会话", use_container_width=True):
        session_manager.clear_current_session()
        st.rerun()

st.title("🧑‍🏫 AI 智能教师")

current_session = session_manager.current_session

if not current_session:
    # 空白态：还没有任何会话
    st.info("点击左侧「➕ 新建会话」开始对话。")
    st.stop()

mode_label = "Agent 自动模式" if agent_mode else "经典对话模式"
st.caption(
    f"当前会话：{current_session} · {subject}教师 · {gender} · "
    f"{personality or '未设置性格'} · {mode_label}"
)

history = session_manager.get_history()
messages = []
if history is not None:
    try:
        messages = history.messages
    except ValueError as exc:
        st.error(str(exc))

for msg in messages:
    st.chat_message("user" if msg.type == "human" else "assistant").write(msg.content)

user_prompt = st.chat_input("请输入你的问题")
if user_prompt:
    if not (subject and gender and personality.strip()):
        st.error("请先在左侧填写：科目、性别、性格")
        st.stop()
    # 用户发送消息时持久化当前设置到磁盘
    session_manager.save_session_settings(subject, gender, personality)
    st.chat_message("user").write(user_prompt)
    with st.chat_message("assistant"):
        placeholder, full_response = st.empty(), ""
        try:
            if agent_mode:
                teaching_state = load_teaching_state(current_session)

                def save_current_state(state):
                    save_teaching_state(current_session, state)

                tools = create_teacher_tools(
                    subject=subject,
                    top_k=top_k,
                    knowledge_base_loader=get_knowledge_base,
                    llm_factory=lambda: get_llm(streaming=False, temperature=0.2),
                    teaching_state=teaching_state,
                    state_saver=save_current_state,
                )
                agent = build_teacher_agent(
                    model=get_llm(streaming=False),
                    tools=tools,
                    subject=subject,
                    gender=gender,
                    personality=personality,
                    teaching_state=teaching_state,
                )
                context_manager = LayeredContextManager(
                    summarizer=lambda prompt: extract_text(
                        get_llm(streaming=False, temperature=0).invoke(prompt)
                    )
                )
                try:
                    with st.spinner("AI 教师正在判断并使用合适的教学工具..."):
                        agent_history = session_manager.get_history()
                        if agent_history is None:
                            raise RuntimeError("当前会话不可用，请新建会话后重试。")
                        agent_result = run_teacher_agent(
                            agent,
                            user_input=user_prompt,
                            history=agent_history,
                            context_manager=context_manager,
                            teaching_state=teaching_state,
                            state_saver=save_current_state,
                        )
                    full_response = agent_result.answer
                    placeholder.markdown(full_response)
                    if agent_result.tool_names:
                        st.caption("已使用工具：" + "、".join(agent_result.tool_names))
                    st.caption(f"Agent 耗时：{agent_result.duration_ms / 1000:.1f} 秒")
                    if agent_result.used_summary:
                        st.caption(
                            f"已使用分层记忆 · 本轮上下文约 "
                            f"{agent_result.context_tokens} tokens"
                        )
                    if agent_result.context_fallback_used:
                        st.caption("摘要暂不可用，本轮已自动采用最近消息窗口。")
                    render_sources(agent_result.sources)
                except Exception:
                    st.warning("Agent 模式暂时不可用，已自动回退为经典回答。")
                    fallback_chain = build_chain(subject, gender, personality, use_context=False)
                    fallback_config = {"configurable": {"session_id": current_session}}
                    for chunk in fallback_chain.stream(
                        {"input": user_prompt, "context": ""}, config=fallback_config
                    ):
                        text = extract_text(chunk)
                        if text:
                            full_response += text
                            placeholder.markdown(full_response)
            else:
                context = ""
                retrieved_with_scores = []
                if use_knowledge:
                    knowledge_base = get_knowledge_base()
                    retrieved_with_scores = knowledge_base.search_with_scores(
                        user_prompt, subject=subject, k=top_k
                    )
                    retrieved = [document for document, _score in retrieved_with_scores]
                    context = knowledge_base.format_context(retrieved)
                    if not context:
                        st.info("知识库没有找到相关片段，将使用普通对话回答。")
                chain = build_chain(subject, gender, personality, use_context=use_knowledge)
                config = {"configurable": {"session_id": current_session}}
                chain_input = {"input": user_prompt, "context": context}
                for chunk in chain.stream(chain_input, config=config):  # type: ignore
                    text = extract_text(chunk)
                    if text:
                        full_response += text
                        placeholder.markdown(full_response)
                if not full_response:
                    st.warning("模型没有返回文本内容，本次消息未写入历史。")
                if retrieved_with_scores:
                    render_sources(knowledge_base.source_records(retrieved_with_scores))
        except (EnvironmentError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"调用模型失败，请检查网络或服务状态：{exc}")
