"""AI 教师会话管理。

这个类只负责会话状态和会话文件操作；具体的 Streamlit 页面控件仍由主程序渲染。
"""

import streamlit as st

from core.memory import (
    create_session,
    delete_session,
    get_session_history,
    list_sessions,
    load_session_meta,
    save_session_meta,
)
from core.teaching_state import clear_teaching_state


class SessionManager:
    """统一管理当前会话以及 sessions 目录中的会话。"""

    CURRENT_KEY = "current_session"
    PICKER_KEY = "session_picker"
    SUBJECT_KEY = "_subject_widget"
    GENDER_KEY = "_gender_widget"
    PERSONALITY_KEY = "_personality_widget"

    def __init__(self, state=None):
        self.state = state if state is not None else st.session_state
        self.ensure_current_session()

    @property
    def current_session(self):
        return self.state.get(self.CURRENT_KEY)

    @property
    def sessions(self):
        return list_sessions()

    def ensure_current_session(self):
        """启动时优先加载最近的已有会话；没有会话时保持空白（None），不自动创建。

        此方法在 __init__ 中调用，早于任何 widget 渲染，可安全写 session_state。
        """
        if self.CURRENT_KEY not in self.state:
            existing = list_sessions()
            if existing:
                self.state[self.CURRENT_KEY] = existing[0]
                self._apply_session_settings(existing[0])
            else:
                self.state[self.CURRENT_KEY] = None
        # 同步 picker 初始值（widget 尚未实例化，可安全写入）
        self.state[self.PICKER_KEY] = self.state.get(self.CURRENT_KEY)

    # ------------------------------------------------------------------
    # 设置持久化（磁盘 .meta.json）
    # ------------------------------------------------------------------

    def _apply_session_settings(self, session_id):
        """从磁盘 meta 文件还原 widget 值。在 on_click/on_change 回调或渲染前调用。"""
        if session_id is None:
            if self.SUBJECT_KEY in self.state:
                del self.state[self.SUBJECT_KEY]
            self.state[self.GENDER_KEY] = "男"
            self.state[self.PERSONALITY_KEY] = ""
            return
        meta = load_session_meta(session_id)
        if meta.get("subject") is not None:
            self.state[self.SUBJECT_KEY] = meta["subject"]
        elif self.SUBJECT_KEY in self.state:
            del self.state[self.SUBJECT_KEY]
        self.state[self.GENDER_KEY] = meta.get("gender", "男")
        self.state[self.PERSONALITY_KEY] = meta.get("personality", "")

    def _save_current_settings(self):
        """把当前 widget 值写到当前会话的磁盘 meta 文件。切换/删除前调用。"""
        sid = self.current_session
        if not sid:
            return
        save_session_meta(sid, {
            "subject": self.state.get(self.SUBJECT_KEY),
            "gender": self.state.get(self.GENDER_KEY, "男"),
            "personality": self.state.get(self.PERSONALITY_KEY, ""),
        })

    def save_session_settings(self, subject, gender, personality):
        """由主程序在用户发送消息时调用，确保不切换时也能持久化设置。"""
        sid = self.current_session
        if not sid:
            return
        save_session_meta(sid, {
            "subject": subject,
            "gender": gender,
            "personality": personality,
        })

    # ------------------------------------------------------------------
    # 会话切换操作（均作为 on_click / on_change 回调使用）
    # ------------------------------------------------------------------

    def new_session(self) -> None:
        """创建并切换到新会话。

        优先复用已存在的最新空会话（无消息），避免重复创建空文件。
        作为按钮 on_click 回调使用，执行时 widget 尚未实例化，可安全写 PICKER_KEY。
        """
        # 找最近创建的空会话（list_sessions 按时间降序排列）
        for sid in self.sessions:
            if not get_session_history(sid).messages:
                # 已有空会话：切过去复用，不新建
                if sid != self.current_session:
                    self._save_current_settings()
                    self.state[self.CURRENT_KEY] = sid
                    self.state[self.PICKER_KEY] = sid
                    self._apply_session_settings(sid)
                return
        # 所有会话都有内容，才真正新建
        self._save_current_settings()
        session_id = create_session()
        self.state[self.CURRENT_KEY] = session_id
        self.state[self.PICKER_KEY] = session_id
        self._apply_session_settings(session_id)

    def load_session(self, session_id):
        """切换到指定会话。不写 PICKER_KEY，避免在 on_change 回调中触发 widget 错误。"""
        if session_id and session_id in self.sessions and session_id != self.current_session:
            self._save_current_settings()
            self.state[self.CURRENT_KEY] = session_id
            self._apply_session_settings(session_id)
        return self.current_session

    def load_selected_session(self) -> None:
        """selectbox on_change 回调：读取下拉框当前值并切换会话。"""
        self.load_session(self.state.get(self.PICKER_KEY))

    def delete_current_session(self):
        """删除当前会话，切换到下一个已有会话；没有则进入空白状态（None）。

        作为按钮 on_click 回调使用，执行时 widget 尚未实例化，可安全写 PICKER_KEY。
        """
        sid = self.current_session
        if not sid:
            return
        delete_session(sid)
        remaining = self.sessions
        next_id = remaining[0] if remaining else None
        self.state[self.CURRENT_KEY] = next_id
        self.state[self.PICKER_KEY] = next_id
        self._apply_session_settings(next_id)

    def clear_current_session(self):
        """清空当前会话消息，但保留会话文件和 ID。"""
        if self.current_session:
            get_session_history(self.current_session).clear()
            clear_teaching_state(self.current_session)

    def get_history(self):
        """返回当前会话的 LangChain 历史对象；无当前会话时返回 None。"""
        if not self.current_session:
            return None
        return get_session_history(self.current_session)
