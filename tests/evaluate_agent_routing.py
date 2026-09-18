"""使用合成工具评测真实模型的 Agent 路由，不读取或发送本地教材。"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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
    {"input": "勾股定理的内容和适用条件是什么？", "expected": ["search_textbook"]},
    {"input": "正弦定理什么时候使用？", "expected": ["search_textbook"]},
    {"input": "导数在几何上表示什么？", "expected": ["search_textbook"]},
    {"input": "古典概型的概率公式是什么？", "expected": ["search_textbook"]},
    {"input": "椭圆的焦点和离心率怎样定义？", "expected": ["search_textbook"]},
    {"input": "等比数列前 n 项和公式怎么推导？", "expected": ["search_textbook"]},
    {"input": "三角函数的最小正周期怎么求？", "expected": ["search_textbook"]},
    {"input": "排列与组合有什么区别？", "expected": ["search_textbook"]},
    {"input": "出一道概率填空题。", "expected": ["generate_exercise"]},
    {"input": "用数列知识测试我一下，不要答案。", "expected": ["generate_exercise"]},
    {"input": "把刚才的题换成选择题。", "expected": ["generate_exercise"]},
    {"input": "给我一道中等难度的三角函数练习。", "expected": ["generate_exercise"]},
    {
        "input": "题目：3x-4=8。我的答案是 x=4。",
        "expected": ["grade_answer"],
    },
    {"input": "我算出来是 x=2，帮我批改。", "expected": ["grade_answer"]},
    {"input": "刚才那题我不会，我的过程是先配方。", "expected": ["grade_answer"]},
    {
        "input": "题目：sin 30° 等于多少？我写的是 1/2。",
        "expected": ["grade_answer"],
    },
    {"input": "我的解题过程写完了，请给分。", "expected": []},
    {"input": "谢谢老师。", "expected": []},
    {"input": "你能做什么？", "expected": []},
    {"input": "我总是粗心，安慰我一下。", "expected": []},
    {"input": "帮我安排一个 25 分钟的番茄钟。", "expected": []},
    {"input": "学了很久，我应该休息多久？", "expected": []},
    {"input": "我今天应该先复习哪一科？", "expected": []},
    {"input": "再见，明天继续。", "expected": []},
]

SCHEMAS = {
    "search_textbook": SearchTextbookInput,
    "generate_exercise": GenerateExerciseInput,
    "grade_answer": GradeAnswerInput,
}


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


def extract_tool_calls(messages) -> list[dict[str, Any]]:
    return [
        call
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def extract_final_answer(messages) -> str:
    answers = [
        str(message.content)
        for message in messages
        if isinstance(message, AIMessage) and not message.tool_calls and message.content
    ]
    return answers[-1] if answers else ""


def validate_tool_arguments(calls: list[dict[str, Any]], expected: list[str]) -> bool:
    if not expected:
        return not calls
    calls_by_name = {call["name"]: call.get("args", {}) for call in calls}
    try:
        return all(
            name in calls_by_name
            and SCHEMAS[name].model_validate(calls_by_name[name]) is not None
            for name in expected
        )
    except (KeyError, TypeError, ValueError):
        return False


def extract_token_usage(messages) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    for message in messages:
        if not isinstance(message, AIMessage) or not message.usage_metadata:
            continue
        input_tokens += int(message.usage_metadata.get("input_tokens", 0))
        output_tokens += int(message.usage_metadata.get("output_tokens", 0))
    return input_tokens, output_tokens


def percentile_95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def evaluate_routes(agent, cases: list[dict[str, Any]]) -> dict[str, Any]:
    details = []
    for case in cases:
        started_at = perf_counter()
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=case["input"])]},
                config={"recursion_limit": 8},
            )
            messages = result["messages"]
            actual = extract_tool_names(messages)
            calls = extract_tool_calls(messages)
            final_answer = extract_final_answer(messages)
            passed = actual == case["expected"]
            arguments_valid = validate_tool_arguments(calls, case["expected"])
            citations = sorted(set(re.findall(r"\[资料\s*(\d+)\]", final_answer)))
            citation_valid = not citations or citations == ["1"]
            citation_present = "search_textbook" not in case["expected"] or citations == ["1"]
            input_tokens, output_tokens = extract_token_usage(messages)
            details.append({
                **case,
                "actual": actual,
                "passed": passed,
                "arguments_valid": arguments_valid,
                "task_completed": passed and bool(final_answer),
                "citation_valid": citation_valid,
                "citation_present": citation_present,
                "answer_faithfulness_proxy": passed and citation_valid and citation_present,
                "duration_ms": round((perf_counter() - started_at) * 1000),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })
        except Exception as exc:
            details.append({
                **case,
                "actual": [],
                "passed": False,
                "arguments_valid": False,
                "task_completed": False,
                "citation_valid": False,
                "citation_present": False,
                "answer_faithfulness_proxy": False,
                "duration_ms": round((perf_counter() - started_at) * 1000),
                "input_tokens": 0,
                "output_tokens": 0,
                "error_type": type(exc).__name__,
            })
    passed_count = sum(detail["passed"] for detail in details)
    tool_cases = [detail for detail in details if detail["expected"]]
    search_cases = [detail for detail in details if "search_textbook" in detail["expected"]]
    latencies = [detail["duration_ms"] for detail in details]
    input_tokens = sum(detail["input_tokens"] for detail in details)
    output_tokens = sum(detail["output_tokens"] for detail in details)
    input_rate = float(os.getenv("EVAL_INPUT_COST_PER_MILLION", "0"))
    output_rate = float(os.getenv("EVAL_OUTPUT_COST_PER_MILLION", "0"))
    estimated_cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total": len(details),
        "passed": passed_count,
        "accuracy": passed_count / len(details) if details else 0.0,
        "metrics": {
            "tool_selection_accuracy": passed_count / len(details) if details else 0.0,
            "parameter_validity_rate": (
                sum(detail["arguments_valid"] for detail in tool_cases) / len(tool_cases)
                if tool_cases
                else 0.0
            ),
            "task_completion_rate": (
                sum(detail["task_completed"] for detail in details) / len(details)
                if details
                else 0.0
            ),
            "citation_correctness": (
                sum(
                    detail["citation_valid"] and detail["citation_present"]
                    for detail in search_cases
                )
                / len(search_cases)
                if search_cases
                else 0.0
            ),
            "answer_faithfulness_proxy": (
                sum(detail["answer_faithfulness_proxy"] for detail in details) / len(details)
                if details
                else 0.0
            ),
            "average_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
            "p95_latency_ms": percentile_95(latencies),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": round(estimated_cost, 6) if input_rate or output_rate else None,
            "cost_rates_per_million": {
                "input": input_rate,
                "output": output_rate,
            },
        },
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
        "## 总体指标",
        "",
        (
            "| 样本 | 工具选择 | 参数有效率 | 任务完成率 | 引用正确率 | "
            "忠实度代理 | 平均延迟 | P95 延迟 |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {report['total']} | {report['metrics']['tool_selection_accuracy']:.1%} | "
            f"{report['metrics']['parameter_validity_rate']:.1%} | "
            f"{report['metrics']['task_completion_rate']:.1%} | "
            f"{report['metrics']['citation_correctness']:.1%} | "
            f"{report['metrics']['answer_faithfulness_proxy']:.1%} | "
            f"{report['metrics']['average_latency_ms'] / 1000:.2f}s | "
            f"{report['metrics']['p95_latency_ms'] / 1000:.2f}s |"
        ),
        "",
        (
            f"Token 用量：输入 {report['metrics']['input_tokens']}，"
            f"输出 {report['metrics']['output_tokens']}。"
        ),
        "",
        "> 引用正确率检查引用编号是否来自合成工具；忠实度是执行轨迹代理指标，"
        "不是 LLM-as-a-Judge 的语义事实评分。成本仅在配置每百万 Token 单价后估算。",
        "",
        "## 明细",
        "",
        "| 输入 | 预期工具 | 实际工具 | 参数 | 耗时 | 结果 |",
        "|---|---|---|---|---:|---|",
    ]
    for detail in report["details"]:
        expected = ", ".join(detail["expected"]) or "不调用工具"
        actual = ", ".join(detail["actual"]) or "不调用工具"
        status = "通过" if detail["passed"] else "失败"
        args_status = "有效" if detail["arguments_valid"] else "无效"
        lines.append(
            f"| {detail['input']} | {expected} | {actual} | {args_status} | "
            f"{detail['duration_ms'] / 1000:.2f}s | {status} |"
        )
    failures = [detail for detail in report["details"] if not detail["passed"]]
    lines.extend(["", "## 失败案例", ""])
    if failures:
        for detail in failures:
            lines.append(
                f"- `{detail['input']}`：预期 {detail['expected']}，实际 {detail['actual']}，"
                f"错误类型 {detail.get('error_type', '路由不一致')}。"
            )
    else:
        lines.append("本轮没有失败案例；历史失败与修复记录见 `docs/failure_cases.md`。")
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
    print(f"工具选择准确率：{report['metrics']['tool_selection_accuracy']:.1%}")
    print(f"参数有效率：{report['metrics']['parameter_validity_rate']:.1%}")
    print(f"任务完成率：{report['metrics']['task_completion_rate']:.1%}")
    print(
        f"平均/P95 延迟：{report['metrics']['average_latency_ms'] / 1000:.2f}s / "
        f"{report['metrics']['p95_latency_ms'] / 1000:.2f}s"
    )
    print(f"报告：{json_path} / {markdown_path}")
    if report["accuracy"] < args.min_accuracy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
