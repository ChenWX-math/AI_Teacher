"""RAG 检索离线评测。

该脚本需要已经构建的 Milvus 集合，并会调用 DashScope Embedding API。
默认评测全部学科，输出 Recall@1、Recall@K、MRR、关键词覆盖率，
同时生成 JSON 和 Markdown 报告。

运行：
    python -m tests.test_rag_retrieval
    python -m tests.test_rag_retrieval --subject 数学
    python -m tests.test_rag_retrieval --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from core.rag import KnowledgeBase


def load_test_questions() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "test_questions.json"
    if not path.exists():
        raise FileNotFoundError(f"测试问题文件不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def color(text: str, code: str) -> str:
    colors = {"green": "92", "red": "91", "yellow": "93", "blue": "94"}
    return f"\033[{colors.get(code, '0')}m{text}\033[0m"


def check_keywords(content: str, keywords: list[str]) -> tuple[int, list[str]]:
    matched, missed = 0, []
    for keyword in keywords:
        if keyword.lower() in content.lower():
            matched += 1
        else:
            missed.append(keyword)
    return matched, missed


def find_expected_rank(sources: list[str], expected_files: list[str]) -> int | None:
    """返回第一个正确来源的 1-based 排名。"""
    for rank, source in enumerate(sources, 1):
        if any(expected in source for expected in expected_files):
            return rank
    return None


def summarize_details(details: list[dict[str, Any]], k: int) -> dict[str, float | int]:
    total = len(details)
    if not total:
        return {
            "total": 0,
            "recall_at_1": 0.0,
            f"recall_at_{k}": 0.0,
            "mrr": 0.0,
            "keyword_coverage": 0.0,
            "errors": 0,
        }

    hit_at_1 = sum(detail.get("expected_rank") == 1 for detail in details)
    hit_at_k = sum(detail.get("expected_rank") is not None for detail in details)
    reciprocal_rank = sum(
        1 / detail["expected_rank"]
        for detail in details
        if detail.get("expected_rank") is not None
    )
    keyword_rates = [float(detail.get("keyword_coverage", 0.0)) for detail in details]
    errors = sum(detail.get("status") == "ERROR" for detail in details)
    return {
        "total": total,
        "recall_at_1": hit_at_1 / total,
        f"recall_at_{k}": hit_at_k / total,
        "mrr": reciprocal_rank / total,
        "keyword_coverage": sum(keyword_rates) / total,
        "errors": errors,
    }


def run_subject_tests(
    kb: KnowledgeBase, subject: str, questions: list[dict[str, Any]], k: int = 3
) -> dict[str, Any]:
    details = []

    for index, question_data in enumerate(questions, 1):
        question = question_data["question"]
        expected_files = question_data["expected_sources"]
        keywords = question_data.get("expected_keywords", [])

        try:
            scored_documents = kb.search_with_scores(question, subject=subject, k=k)
        except Exception as exc:
            details.append({
                "index": index,
                "question": question,
                "status": "ERROR",
                "error": str(exc),
                "expected_rank": None,
                "keyword_coverage": 0.0,
            })
            continue

        documents = [document for document, _score in scored_documents]
        sources = [str(document.metadata.get("source", "未知")) for document in documents]
        scores = [round(float(score), 6) for _document, score in scored_documents]
        expected_rank = find_expected_rank(sources, expected_files)
        content = "\n".join(document.page_content for document in documents)
        matched, missed = check_keywords(content, keywords)
        keyword_coverage = matched / len(keywords) if keywords else 1.0

        if not documents:
            status = "FAIL"
            reason = "未检索到任何结果"
        elif expected_rank is None:
            status = "FAIL"
            reason = "Top-K 中没有预期来源"
        elif keyword_coverage < 0.5:
            status = "WEAK"
            reason = "正确来源已召回，但关键信息覆盖不足 50%"
        else:
            status = "PASS"
            reason = ""

        details.append({
            "index": index,
            "question": question,
            "status": status,
            "reason": reason,
            "expected_sources": expected_files,
            "sources": sources,
            "scores": scores,
            "expected_rank": expected_rank,
            "keywords_matched": matched,
            "keywords_total": len(keywords),
            "keywords_missed": missed,
            "keyword_coverage": round(keyword_coverage, 6),
        })

    return {
        "subject": subject,
        "k": k,
        "metrics": summarize_details(details, k),
        "details": details,
    }


def print_subject_results(results: dict[str, Any]) -> None:
    metrics = results["metrics"]
    k = results["k"]
    recall_k = metrics[f"recall_at_{k}"]
    print(f"\n{'=' * 72}")
    print(
        f"  📚 {results['subject']} | n={metrics['total']} | "
        f"R@1={metrics['recall_at_1']:.1%} | R@{k}={recall_k:.1%} | "
        f"MRR={metrics['mrr']:.3f} | 关键词={metrics['keyword_coverage']:.1%}"
    )
    print(f"{'=' * 72}")

    for detail in results["details"]:
        status = detail["status"]
        icon = {
            "PASS": color("✓", "green"),
            "WEAK": color("△", "yellow"),
            "FAIL": color("✗", "red"),
            "ERROR": color("!", "red"),
        }.get(status, "?")
        print(f"\n  [{detail['index']}] {icon} {detail['question']}")
        if status == "ERROR":
            print(f"      错误：{detail['error']}")
            continue
        print(
            f"      正确来源排名：{detail['expected_rank'] or '未命中'} | "
            f"关键词：{detail['keywords_matched']}/{detail['keywords_total']}"
        )
        for source, score in zip(detail["sources"], detail["scores"], strict=True):
            print(f"      {score:.3f}  {source}")
        if detail["reason"]:
            print(f"      说明：{detail['reason']}")


def combine_results(subject_results: list[dict[str, Any]], k: int) -> dict[str, Any]:
    all_details = [
        detail
        for result in subject_results
        for detail in result["details"]
    ]
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "k": k,
        "metrics": summarize_details(all_details, k),
        "subjects": subject_results,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    k = report["k"]
    lines = [
        "# AI 智能教师 RAG 检索评测",
        "",
        f"生成时间：{report['generated_at']}",
        "",
        "## 总体指标",
        "",
        f"| 样本数 | Recall@1 | Recall@{k} | MRR | 关键词覆盖率 | 错误数 |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {metrics['total']} | {metrics['recall_at_1']:.1%} | "
            f"{metrics[f'recall_at_{k}']:.1%} | {metrics['mrr']:.3f} | "
            f"{metrics['keyword_coverage']:.1%} | {metrics['errors']} |"
        ),
        "",
        "## 分学科指标",
        "",
        f"| 学科 | 样本数 | Recall@1 | Recall@{k} | MRR | 关键词覆盖率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in report["subjects"]:
        subject_metrics = result["metrics"]
        lines.append(
            f"| {result['subject']} | {subject_metrics['total']} | "
            f"{subject_metrics['recall_at_1']:.1%} | "
            f"{subject_metrics[f'recall_at_{k}']:.1%} | "
            f"{subject_metrics['mrr']:.3f} | "
            f"{subject_metrics['keyword_coverage']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rag_evaluation.json"
    markdown_path = output_dir / "rag_evaluation.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def run_interactive_mode(kb: KnowledgeBase, questions_data: dict[str, Any]) -> None:
    print("\n📋 交互测试模式：逐条输入问题，实时查看检索结果。\n")
    subjects = list(questions_data["subjects"].keys())
    print(f"可用科目：{', '.join(subjects)}")
    subject_input = input("\n请输入科目：").strip()
    subject = subject_input if subject_input in subjects else None
    k_input = input("检索条数 (默认 3)：").strip()
    k = int(k_input) if k_input else 3

    while True:
        question = input("\n问题（输入 q 退出）：").strip()
        if question.lower() == "q":
            break
        if not question:
            continue
        results = kb.search_with_scores(question, subject=subject, k=k)
        for source in kb.source_records(results):
            print(f"{source['score']:.3f} | {source['source']} | {source['preview']}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="AI 智能教师 RAG 检索评测")
    parser.add_argument("--subject", help="只评测指定科目")
    parser.add_argument("--interactive", action="store_true", help="交互检索模式")
    parser.add_argument("-k", type=int, default=3, help="每条查询返回结果数")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation_results"), help="报告输出目录"
    )
    parser.add_argument(
        "--min-recall", type=float, default=0.8, help="Recall@K 最低通过阈值"
    )
    args = parser.parse_args()

    questions_data = load_test_questions()
    print("⏳ 正在加载知识库...")
    try:
        kb = KnowledgeBase().load()
    except Exception as exc:
        print(f"\n❌ 知识库加载失败：{exc}")
        print("   请先运行 python -m core.rag --build 构建知识库。")
        raise SystemExit(1) from exc

    if args.interactive:
        run_interactive_mode(kb, questions_data)
        return

    subjects = [args.subject] if args.subject else list(questions_data["subjects"])
    results = []
    for subject in subjects:
        questions = questions_data["subjects"].get(subject)
        if not questions:
            parser.error(f"科目 {subject!r} 不在评测集中")
        subject_result = run_subject_tests(kb, subject, questions, k=args.k)
        print_subject_results(subject_result)
        results.append(subject_result)

    report = combine_results(results, args.k)
    json_path, markdown_path = write_reports(report, args.output_dir)
    metrics = report["metrics"]
    recall_k = metrics[f"recall_at_{args.k}"]
    print(f"\n总计：R@1={metrics['recall_at_1']:.1%}，R@{args.k}={recall_k:.1%}，")
    print(f"MRR={metrics['mrr']:.3f}，关键词覆盖率={metrics['keyword_coverage']:.1%}")
    print(f"报告：{json_path} / {markdown_path}")

    if metrics["errors"] or recall_k < args.min_recall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
