"""
提示词库管理模块

负责从 subject_prompts.json 中读取 9 个高中科目的提示词，
根据「科目 + 性别 + 性格」三个变量，组装出完整的系统提示词。

设计思路：
    - 提示词文案全部放在 JSON 文件里（数据），本类只负责读取与拼接（逻辑）。
    - 新增/修改科目，只需改 JSON，不用改这里的代码。
    - 三个变量：科目(subject)、性别(gender)、性格(personality)。
"""

import json
import os


class PromptManager:
    """提示词库管理器：加载 JSON 并按需生成系统提示词。"""

    def __init__(self, json_path=None):
        """
        初始化：加载提示词 JSON 文件。

        Args:
            json_path (str, optional): JSON 文件路径。默认使用本文件同目录下的
                subject_prompts.json。
        """
        if json_path is None:
            # 与本文件同目录下的提示词库文件
            json_path = os.path.join(os.path.dirname(__file__), "subject_prompts.json")

        with open(json_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        self._shared = self._data["_shared"]       # 通用部分（人设模板、守则、输出风格）
        self._subjects = self._data["subjects"]    # 各科目专属内容

    def list_subjects(self):
        """返回所有可用科目名列表，例如 ["语文", "数学", ...]"""
        return list(self._subjects.keys())

    def has_subject(self, subject):
        """判断某个科目是否存在。"""
        return subject in self._subjects

    def build_system_prompt(self, subject, gender, personality):
        """
        根据三个变量生成完整系统提示词。

        Args:
            subject (str): 科目，如 "数学"。
            gender (str): 性别，如 "男" / "女"。
            personality (str): 性格/教学风格，如 "温柔耐心、擅长用生活例子讲解"。

        Returns:
            str: 组装好的系统提示词（可直接作为 system message 的 content）。
        """
        # 1. 人设（填入三个变量）
        persona = self._shared["persona_template"].format(
            subject=subject, gender=gender, personality=personality
        )

        lines = [persona, ""]

        # 2. 通用教学守则
        lines.append("【教学守则 · 所有科目通用】")
        for i, rule in enumerate(self._shared["general_rules"], 1):
            lines.append(f"{i}. {rule}")
        lines.append("")

        # 3. 学科专属内容（科目不存在时退化为通用老师）
        subj = self._subjects.get(subject)
        if subj:
            lines.append(f"【学科定位】{subject}：{subj['focus']}")
            lines.append("")

            lines.append("【教学侧重点】")
            for point in subj["teaching_points"]:
                lines.append(f"- {point}")
            lines.append("")

            lines.append("【常见题型】")
            for q in subj["question_types"]:
                lines.append(f"- {q}")
            lines.append("")

            lines.append("【本学科讲解方法】")
            for i, method in enumerate(subj["method_guide"], 1):
                lines.append(f"{i}. {method}")
            lines.append("")
        else:
            lines.append("【学科定位】请以一位负责任的老师身份，就学生提出的问题给出专业、清晰的辅导。")
            lines.append("")

        # 4. 输出风格
        lines.append("【输出风格】")
        for style in self._shared["output_style"]:
            lines.append(f"- {style}")

        return "\n".join(lines)


if __name__ == "__main__":
    # 简单自测：打印一个示例提示词
    pm = PromptManager()
    print("已加载科目：", pm.list_subjects())
    print("\n" + "=" * 60)
    print(pm.build_system_prompt("数学", "女", "温柔耐心、擅长用生活例子讲解"))
