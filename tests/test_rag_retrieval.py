"""RAG 检索精度自动化测试。

对每个科目逐一发起检索请求，检查：
1. 是否能检索到结果（数量 > 0）
2. 检索结果是否来自预期的资料文件
3. 检索内容是否包含预期关键词（人工辅助判断）

运行：
    cd AI_Teacher
    python -m tests.test_rag_retrieval              # 测试所有科目
    python -m tests.test_rag_retrieval --subject 数学  # 只测数学
    python -m tests.test_rag_retrieval --interactive     # 逐条交互模式
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from core.rag import KnowledgeBase


def load_test_questions() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "test_questions.json"
    if not path.exists():
        raise FileNotFoundError(f"测试问题文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def color(text: str, code: str) -> str:
    """简单的 ANSI 着色（PowerShell 兼容）。"""
    colors = {"green": "92", "red": "91", "yellow": "93", "blue": "94"}
    return f"\033[{colors.get(code, '0')}m{text}\033[0m"


def check_keywords(content: str, keywords: list[str]) -> tuple[int, list[str]]:
    matched, missed = 0, []
    for kw in keywords:
        if kw.lower() in content.lower():
            matched += 1
        else:
            missed.append(kw)
    return matched, missed


def run_subject_tests(kb: KnowledgeBase, subject: str, questions: list[dict],
                      k: int = 3) -> dict[str, Any]:
    results = {"subject": subject, "total": len(questions), "passed": 0, "details": []}

    for i, q in enumerate(questions):
        question = q["question"]
        expected_files = q["expected_sources"]
        keywords = q.get("expected_keywords", [])

        try:
            docs = kb.search(question, subject=subject, k=k)
        except Exception as exc:
            results["details"].append({
                "index": i + 1, "question": question, "status": "ERROR",
                "error": str(exc),
            })
            continue

        if not docs:
            results["details"].append({
                "index": i + 1, "question": question, "status": "FAIL",
                "reason": "未检索到任何结果",
            })
            continue

        sources = [doc.metadata.get("source", "未知") for doc in docs]
        hit_expected = any(
            any(expected in src for expected in expected_files) for src in sources
        )

        all_content = "\n".join(doc.page_content for doc in docs)
        kw_matched, kw_missed = check_keywords(all_content, keywords)
        status = "PASS" if hit_expected else "WEAK"
        if hit_expected:
            results["passed"] += 1

        results["details"].append({
            "index": i + 1,
            "question": question,
            "status": status,
            "sources": sources,
            "hit_expected": hit_expected,
            "keywords_matched": f"{kw_matched}/{len(keywords)}",
            "keywords_missed": kw_missed,
        })

    return results


def print_subject_results(results: dict[str, Any]) -> None:
    subject = results["subject"]
    total = results["total"]
    passed = results["passed"]
    rate = passed / total * 100 if total else 0

    print(f"\n{'=' * 60}")
    print(f"  📚 科目：{subject}  |  通过率：{passed}/{total} ({rate:.0f}%)")
    print(f"{'=' * 60}")

    for detail in results["details"]:
        idx = detail["index"]
        q = detail["question"]
        status = detail["status"]

        icon = {"PASS": color("✓", "green"),
                "WEAK": color("△", "yellow"),
                "FAIL": color("✗", "red"),
                "ERROR": color("!", "red")}.get(status, "?")

        print(f"\n  [{idx}] {icon} {q}")

        if status == "ERROR":
            print(f"      ❌ 错误：{detail['error']}")
            continue
        if status == "FAIL":
            print(f"      ❌ {detail['reason']}")
            continue

        sources = detail["sources"]
        for s in sources:
            hit = "✓" if detail["hit_expected"] else "?"
            print(f"      来源 {hit}：{s}")

        kw_info = detail["keywords_matched"]
        if detail.get("keywords_missed"):
            missed = ", ".join(detail["keywords_missed"])
            print(f"      关键词 {kw_info} | 未命中：{missed}")
        else:
            print(f"      关键词 {kw_info}（全部命中）")


def run_interactive_mode(kb: KnowledgeBase, questions_data: dict[str, Any]) -> None:
    print("\n📋 交互测试模式：逐条输入问题，实时查看检索结果。\n")
    subjects = list(questions_data["subjects"].keys())
    print(f"可用科目：{', '.join(subjects)}")

    subject = input("\n请输入科目：").strip()
    if subject not in subjects:
        print(f"科目 '{subject}' 不在测试集中，将不限制科目。")
        subject = None

    k_str = input("检索条数 (默认 3)：").strip()
    k = int(k_str) if k_str else 3

    while True:
        q = input("\n问题（输入 q 退出）：").strip()
        if q.lower() == "q":
            break
        if not q:
            continue
        try:
            docs = kb.search(q, subject=subject, k=k)
            print(f"\n检索到 {len(docs)} 条：")
            print(kb.format_context(docs))
        except Exception as exc:
            print(f"❌ 错误：{exc}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="RAG 检索精度测试")
    parser.add_argument("--subject", help="指定科目，不指定则测试所有科目")
    parser.add_argument("--interactive", action="store_true", help="交互模式：手动输入问题")
    parser.add_argument("-k", type=int, default=3, help="每条查询返回的结果数（默认 3）")
    args = parser.parse_args()

    questions_data = load_test_questions()
    all_subjects = list(questions_data["subjects"].keys())

    print("⏳ 正在加载知识库...")
    try:
        kb = KnowledgeBase().load()
    except Exception as exc:
        print(f"\n❌ 知识库加载失败：{exc}")
        print("   请先运行 python -m core.rag --build 构建知识库。")
        sys.exit(1)

    print("✅ 知识库加载完成。\n")

    if args.interactive:
        run_interactive_mode(kb, questions_data)
        return

    subjects_to_test = [args.subject] if args.subject else all_subjects

    overall = {"total": 0, "passed": 0}
    for subject in subjects_to_test:
        questions = questions_data["subjects"].get(subject)
        if not questions:
            print(f"⚠️  科目 '{subject}' 没有测试问题，跳过。")
            continue
        results = run_subject_tests(kb, subject, questions, k=args.k)
        print_subject_results(results)
        overall["total"] += results["total"]
        overall["passed"] += results["passed"]

    print(f"\n{'=' * 60}")
    rate = overall["passed"] / overall["total"] * 100 if overall["total"] else 0
    print(f"  🏁 总计：{overall['passed']}/{overall['total']} 通过（{rate:.0f}%）")
    print(f"{'=' * 60}")

    if rate < 60:
        print("\n  ⚠️  通过率偏低，建议：")
        print("     1. 检查知识库是否已构建（python -m core.rag --build）")
        print("     2. 调整 chunk_size/chunk_overlap 参数")
        print("     3. 检查资料文件的编码是否为 UTF-8")


if __name__ == "__main__":
    main()
