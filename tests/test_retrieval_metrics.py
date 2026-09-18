"""检索评测指标与报告测试，不访问外部服务。"""

import json

from tests.test_rag_retrieval import (
    combine_results,
    find_expected_rank,
    render_markdown_report,
    summarize_details,
    write_reports,
)


def test_find_expected_rank():
    sources = ["数学/数列.txt", "数学/解析几何.txt"]

    assert find_expected_rank(sources, ["解析几何.txt"]) == 2
    assert find_expected_rank(sources, ["导数.txt"]) is None


def test_summarize_details_calculates_retrieval_metrics():
    details = [
        {"expected_rank": 1, "keyword_coverage": 1.0, "status": "PASS"},
        {"expected_rank": 2, "keyword_coverage": 0.5, "status": "PASS"},
        {"expected_rank": None, "keyword_coverage": 0.0, "status": "FAIL"},
    ]

    metrics = summarize_details(details, k=3)

    assert metrics["recall_at_1"] == 1 / 3
    assert metrics["recall_at_3"] == 2 / 3
    assert metrics["mrr"] == 0.5
    assert metrics["keyword_coverage"] == 0.5


def test_report_can_be_rendered_and_written(tmp_path):
    subject_result = {
        "subject": "数学",
        "k": 3,
        "metrics": {
            "total": 1,
            "recall_at_1": 1.0,
            "recall_at_3": 1.0,
            "mrr": 1.0,
            "keyword_coverage": 0.75,
            "errors": 0,
        },
        "details": [{"expected_rank": 1, "keyword_coverage": 0.75, "status": "PASS"}],
    }
    report = combine_results([subject_result], k=3)

    markdown = render_markdown_report(report)
    json_path, markdown_path = write_reports(report, tmp_path)

    assert "Recall@1" in markdown
    assert "数学" in markdown
    assert json.loads(json_path.read_text(encoding="utf-8"))["metrics"]["mrr"] == 1.0
    assert markdown_path.read_text(encoding="utf-8") == markdown
