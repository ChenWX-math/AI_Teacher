# 典型失败案例与回归修复

这些案例来自真实开发与评测，不是为了展示而虚构的理想结果。每次修复都沉淀为规则、测试或评测样本。

| 失败现象 | 根因 | 修复 | 回归证据 |
|---|---|---|---|
| 第一版 Agent 路由只有 6/9 | Prompt 只描述工具能力，没有明确互斥优先级 | 增加“批改 > 出题 > 教材检索 > 普通交流”的强规则 | `tests/evaluate_agent_routing.py` |
| DeepSeek 自由 JSON 输出偶尔改字段名 | JSON 模式没有强制遵守 Pydantic Schema | 改用 Function Calling + `strict=True` | `tests/test_teacher_tools.py` |
| “我的答案是……”缺少完整题目 | 题目只存在聊天文本中，工具无法稳定关联 | 当前练习与隐藏参考答案写入独立教学状态 | `tests/test_teaching_state.py` |
| 长对话可能超过上下文窗口 | 每轮都提交完整历史 | 较早摘要 + 最近原文 + 结构化状态；摘要失败回退滑动窗口 | `tests/test_context_manager.py` |
| 同一会话切换学科可能沿用旧题 | 教学状态没有检查科目变化 | 科目变化时清空当前知识点、练习和最近批改 | `test_switching_subject_clears_old_exercise` |
| 状态落盘失败会让已生成内容看起来失败 | 状态增强与主要回答处于同一个异常分支 | 状态保存改为非阻断，并记录错误日志 | `test_state_save_failure_does_not_discard_generated_exercise` |
| 36 条评测首轮为 34/36 | “不会 + 部分过程”被误判成只要提示；“写完了”却没附内容的样本标签不合理 | 明确部分过程优先批改；无实际内容时澄清，并校准评测标签 | `evaluation_results/agent_routing.md` |

## 如何继续使用失败案例

1. 从 JSONL 日志中按 `request_id` 定位失败请求。
2. 判断失败属于检索、工具路由、参数、模型回答还是状态管理。
3. 删除敏感原文，只保留最小合成复现输入。
4. 先把复现输入加入 pytest 或离线评测，再修改 Prompt、Schema 或代码。
5. 同时观察质量、平均/P95 延迟和 Token 变化，避免只修正确率却显著增加成本。
