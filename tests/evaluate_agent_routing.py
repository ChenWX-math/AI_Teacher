"""使用合成工具评测真实模型的 Agent 路由，不读取或发送本地教材。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from core.llm import get_llm
from core.teacher_agent import build_teacher_agent
from core.teacher_tools import (
    GenerateExerciseInput,
    GradeAnswerInput,
    SearchTextbookInput,
)
from core.teaching_state import ExerciseState, TeachingState

ROUTING_CASES = [
    {"input": "二次函数的顶点坐标公式是什么？", "expected": ["search_textbook"]},
    {"input": "等差数列的通项公式怎么写？", "expected": ["search_textbook"]},
    {"input": "请根据二次函数给我出一道基础题，先不要答案。", "expected": ["generate_exercise"]},
    {"input": "给我来一道稍难的函数选择题，并附答案。", "expected": ["generate_exercise"]},
    {
        "input": "题目：求 y=x²-2x+1 的顶点。我的答案是 (1,0)，请批改。",
        "expected": ["grade_answer"],
    },
    {
        "input": "题目：2x+1=5。我的解答是 x=3，帮我看看哪里错了。",
        "expected": ["grade_answer"],
    },
    {"input": "你好，你是谁？", "expected": []},
    {"input": "今天学习有点累，鼓励我一下。", "expected": []},
    {"input": "帮我制定一个今晚复习数学的简单安排。", "expected": []},
    {"input": "再来一道。", "expected": ["generate_exercise"]},
    {"input": "这道太简单了，难一点。", "expected": ["generate_exercise"]},
    {"input": "我的答案是 (1, 0)。", "expected": ["grade_answer"]},
]


def create_stub_tools():
    def search_textbook(query: str):
        return f"[资料 1] 合成教材结果：{query}"

    def generate_exercise(
        topic: str,
        difficulty: str = "基础",
        question_type: str = "解答题",
        include_answer: bool = False,
    ):
        answer = "；答案：合成答案" if include_answer else ""
        return f"合成练习：{topic}，{difficulty}，{question_type}{answer}"

    def grade_answer(student_answer: str, question: str = ""):
        return f"合成批改：题目={question}；学生答案={student_answer}"

    return [
        StructuredTool.from_function(
            func=search_textbook,
            name="search_textbook",
            description="数学概念、定义、公式、解题方法和事实类问题必须使用。",
            args_schema=SearchTextbookInput,
        ),
        StructuredTool.from_function(
            func=generate_exercise,
            name="generate_exercise",
            description="学生要求出题、练习或改变题目难度时必须使用。",
            args_schema=GenerateExerciseInput,
        ),
        StructuredTool.from_function(
            func=grade_answer,
            name="grade_answer",
            description="学生提交答案、计算结果、解题过程或要求批改时必须使用。",
            args_schema=GradeAnswerInput,
        ),
    ]


def extract_tool_names(messages) -> list[str]:
    names = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name and message.name not in names:
            names.append(message.name)
    return names


def evaluate_routes(agent, cases: list[dict[str, Any]]) -> dict[str, Any]:
    details = []
    for case in cases:
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=case["input"])]},
                config={"recursion_limit": 8},
            )
            actual = extract_tool_names(result["messages"])
            passed = actual == case["expected"]
            details.append({**case, "actual": actual, "passed": passed})
        except Exception as exc:
            details.append({**case, "actual": [], "passed": False, "error": str(exc)})
    passed_count = sum(detail["passed"] for detail in details)
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total": len(details),
        "passed": passed_count,
        "accuracy": passed_count / len(details) if details else 0.0,
        "details": details,
    }


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "agent_routing.json"
    markdown_path = output_dir / "agent_routing.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# AI 智能教师 Agent 路由评测",
        "",
        f"生成时间：{report['generated_at']}",
        "",
        f"工具选择准确率：**{report['passed']}/{report['total']} ({report['accuracy']:.1%})**",
        "",
        "| 输入 | 预期工具 | 实际工具 | 结果 |",
        "|---|---|---|---|",
    ]
    for detail in report["details"]:
        expected = ", ".join(detail["expected"]) or "不调用工具"
        actual = ", ".join(detail["actual"]) or "不调用工具"
        status = "通过" if detail["passed"] else "失败"
        lines.append(f"| {detail['input']} | {expected} | {actual} | {status} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="AI 教师 Agent 工具路由评测")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument("--min-accuracy", type=float, default=0.8)
    args = parser.parse_args()

    agent = build_teacher_agent(
        model=get_llm(streaming=False, temperature=0.0),
        tools=create_stub_tools(),
        subject="数学",
        gender="女",
        personality="耐心、讲解清晰",
        teaching_state=TeachingState(
            subject="数学",
            current_topic="二次函数",
            current_exercise=ExerciseState(
                topic="二次函数",
                difficulty="基础",
                question_type="解答题",
                question="求 y=x²-2x+1 的顶点。",
                reference_answer="(1, 0)",
                explanation="配方得到 y=(x-1)²。",
            ),
        ),
    )
    report = evaluate_routes(agent, ROUTING_CASES)
    json_path, markdown_path = write_report(report, args.output_dir)
    for detail in report["details"]:
        print(
            f"{'PASS' if detail['passed'] else 'FAIL'} | {detail['input']} | "
            f"expected={detail['expected']} actual={detail['actual']}"
        )
    print(f"准确率：{report['accuracy']:.1%}")
    print(f"报告：{json_path} / {markdown_path}")
    if report["accuracy"] < args.min_accuracy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
