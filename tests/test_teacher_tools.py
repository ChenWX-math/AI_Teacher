"""AI 教师工具测试，不访问模型或向量数据库。"""

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

from core.teacher_tools import ExerciseResult, GradeResult, create_teacher_tools
from core.teaching_state import TeachingState


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
            next_step="尝试一道中等难度的同类题。",
        )


class FakeLLM:
    def with_structured_output(self, schema, **_kwargs):
        return FakeStructuredModel(schema)


def build_tools(teaching_state=None, state_saver=None):
    return create_teacher_tools(
        subject="数学",
        top_k=2,
        knowledge_base_loader=FakeKnowledgeBase,
        llm_factory=FakeLLM,
        teaching_state=teaching_state,
        state_saver=state_saver,
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


def test_generate_then_grade_current_exercise_updates_persisted_state():
    state = TeachingState()
    saved = []
    exercise_tool, grade_tool = build_tools(
        teaching_state=state,
        state_saver=lambda current: saved.append(current.model_copy(deep=True)),
    )[1:]

    exercise_result = invoke_tool(exercise_tool, {
        "topic": "二次函数",
        "difficulty": "基础",
        "question_type": "解答题",
        "include_answer": False,
    })
    grade_result = invoke_tool(grade_tool, {"student_answer": "(1, 0)"})

    assert state.current_exercise.question == "求 y=x²-2x+1 的顶点。"
    assert state.current_exercise.reference_answer == "(1, 0)"
    assert "(1, 0)" not in exercise_result.content
    assert state.last_grade.score == 100
    assert grade_result.artifact["grade"]["correct"] is True
    assert len(saved) == 2


def test_grade_without_question_or_current_exercise_requests_question():
    grade_tool = build_tools(teaching_state=TeachingState())[2]

    result = invoke_tool(grade_tool, {"student_answer": "42"})

    assert result.artifact["needs_question"] is True


def test_switching_subject_clears_old_exercise():
    state = TeachingState(subject="数学")
    exercise_tool = build_tools(teaching_state=state)[1]
    invoke_tool(exercise_tool, {
        "topic": "二次函数",
        "difficulty": "基础",
        "question_type": "解答题",
        "include_answer": False,
    })

    create_teacher_tools(
        subject="物理",
        top_k=2,
        knowledge_base_loader=FakeKnowledgeBase,
        llm_factory=FakeLLM,
        teaching_state=state,
    )

    assert state.subject == "物理"
    assert state.current_topic is None
    assert state.current_exercise is None
    assert state.last_grade is None


def test_state_save_failure_does_not_discard_generated_exercise():
    def fail_to_save(_state):
        raise OSError("disk full")

    exercise_tool = build_tools(
        teaching_state=TeachingState(),
        state_saver=fail_to_save,
    )[1]

    result = invoke_tool(exercise_tool, {
        "topic": "二次函数",
        "difficulty": "基础",
        "question_type": "解答题",
        "include_answer": False,
    })

    assert "求 y=x²-2x+1 的顶点" in result.content
    assert "error_type" not in result.artifact
