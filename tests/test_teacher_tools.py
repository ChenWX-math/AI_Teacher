"""AI 教师工具测试，不访问模型或向量数据库。"""

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

from core.teacher_tools import ExerciseResult, GradeResult, create_teacher_tools


class FakeKnowledgeBase:
    def search_with_scores(self, query, subject, k):
        assert query == "顶点公式"
        assert subject == "数学"
        assert k == 2
        document = Document(
            page_content="顶点坐标为 (-b/(2a), (4ac-b²)/(4a))",
            metadata={"source": "数学/函数.txt", "chunk_index": 1},
        )
        return [(document, 0.91)]

    format_context = staticmethod(lambda documents: f"[资料 1]\n{documents[0].page_content}")
    source_records = staticmethod(
        lambda results: [{
            "label": "资料 1",
            "source": "数学/函数.txt",
            "chunk_index": 1,
            "score": results[0][1],
            "preview": "顶点坐标",
        }]
    )


class FakeStructuredModel:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, _prompt):
        if self.schema is ExerciseResult:
            return ExerciseResult(
                topic="二次函数",
                difficulty="基础",
                question_type="解答题",
                question="求 y=x²-2x+1 的顶点。",
                reference_answer="(1, 0)",
                explanation="配方得到 y=(x-1)²。",
            )
        return GradeResult(
            correct=True,
            score=100,
            feedback="计算正确。",
            error_reason="",
            hint="继续保持。",
            reference_solution="配方得到顶点 (1, 0)。",
        )


class FakeLLM:
    def with_structured_output(self, schema, **_kwargs):
        return FakeStructuredModel(schema)


def build_tools():
    return create_teacher_tools(
        subject="数学",
        top_k=2,
        knowledge_base_loader=FakeKnowledgeBase,
        llm_factory=FakeLLM,
    )


def invoke_tool(tool, args):
    return tool.invoke({
        "name": tool.name,
        "args": args,
        "id": f"call-{tool.name}",
        "type": "tool_call",
    })


def test_search_textbook_returns_context_and_source_artifact():
    search_tool = build_tools()[0]

    result = invoke_tool(search_tool, {"query": "顶点公式"})

    assert isinstance(result, ToolMessage)
    assert "[资料 1]" in result.content
    assert result.artifact["sources"][0]["score"] == 0.91


def test_generate_exercise_can_hide_reference_answer():
    exercise_tool = build_tools()[1]

    result = invoke_tool(exercise_tool, {
        "topic": "二次函数",
        "difficulty": "基础",
        "question_type": "解答题",
        "include_answer": False,
    })

    assert "求 y=x²-2x+1 的顶点" in result.content
    assert "(1, 0)" not in result.content
    assert result.artifact["exercise"]["reference_answer"] == "(1, 0)"


def test_grade_answer_returns_structured_feedback():
    grade_tool = build_tools()[2]

    result = invoke_tool(grade_tool, {
        "question": "求 y=x²-2x+1 的顶点。",
        "student_answer": "(1, 0)",
    })

    assert "100 分" in result.content
    assert result.artifact["grade"]["correct"] is True
