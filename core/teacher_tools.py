"""AI 教师可调用的教材检索、练习生成和答案批改工具。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from core.rag import KnowledgeBase


class SearchTextbookInput(BaseModel):
    query: str = Field(description="需要从当前学科教材中查询的具体问题或知识点")


class GenerateExerciseInput(BaseModel):
    topic: str = Field(description="练习对应的知识点")
    difficulty: str = Field(default="基础", description="基础、中等或提高")
    question_type: str = Field(default="解答题", description="选择题、填空题或解答题")
    include_answer: bool = Field(default=False, description="是否在本次回复中展示答案和解析")


class GradeAnswerInput(BaseModel):
    question: str = Field(description="需要批改的完整题目")
    student_answer: str = Field(description="学生提交的答案或解题过程")


class ExerciseResult(BaseModel):
    topic: str
    difficulty: str
    question_type: str
    question: str
    reference_answer: str
    explanation: str


class GradeResult(BaseModel):
    correct: bool
    score: int = Field(ge=0, le=100)
    feedback: str
    error_reason: str
    hint: str
    reference_solution: str


def _invoke_structured(
    llm_factory: Callable[[], BaseChatModel], schema: type[BaseModel], prompt: str
) -> BaseModel:
    llm = llm_factory()
    # DeepSeek 的自由 JSON 模式可能自行改字段名；Function Calling 会把
    # Pydantic Schema 作为工具参数约束，结构化输出更稳定。
    structured_llm = llm.with_structured_output(
        schema,
        method="function_calling",
        strict=True,
    )
    return structured_llm.invoke(prompt)


def create_teacher_tools(
    *,
    subject: str,
    top_k: int,
    knowledge_base_loader: Callable[[], KnowledgeBase],
    llm_factory: Callable[[], BaseChatModel],
) -> list[BaseTool]:
    """创建绑定当前学科的三个工具；知识库只在实际检索时加载。"""

    def search_textbook(query: str) -> tuple[str, dict[str, Any]]:
        """检索当前学科教材，用于事实、概念、公式和教材依据类问题。"""
        try:
            knowledge_base = knowledge_base_loader()
            results = knowledge_base.search_with_scores(query, subject=subject, k=top_k)
            if not results:
                return "当前教材没有找到足够相关的资料。", {"sources": []}
            documents = [document for document, _score in results]
            return knowledge_base.format_context(documents), {
                "sources": knowledge_base.source_records(results)
            }
        except Exception as exc:
            error_type = type(exc).__name__
            return (
                f"教材检索暂时失败，请稍后重试。错误类型：{error_type}",
                {"sources": [], "error_type": error_type},
            )

    def generate_exercise(
        topic: str,
        difficulty: str = "基础",
        question_type: str = "解答题",
        include_answer: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """根据学科、知识点、难度和题型生成一道练习题。"""
        prompt = f"""你是高中{subject}教师。请生成一道可独立作答的练习题。
知识点：{topic}
难度：{difficulty}
题型：{question_type}
请确保题目条件完整、答案唯一或评分标准明确，并严格输出要求的 JSON 字段。"""
        try:
            result = _invoke_structured(llm_factory, ExerciseResult, prompt)
            exercise = ExerciseResult.model_validate(result).model_dump()
        except Exception as exc:
            error_type = type(exc).__name__
            return (
                f"练习生成暂时失败，请稍后重试。错误类型：{error_type}",
                {"error_type": error_type},
            )

        content = f"题目：{exercise['question']}"
        if include_answer:
            content += (
                f"\n\n参考答案：{exercise['reference_answer']}"
                f"\n\n解析：{exercise['explanation']}"
            )
        else:
            content += "\n\n本次先不展示答案和解析，等待学生作答。"
        return content, {"exercise": exercise, "include_answer": include_answer}

    def grade_answer(question: str, student_answer: str) -> tuple[str, dict[str, Any]]:
        """批改学生对指定题目的答案，给出得分、错误原因、提示和参考解法。"""
        prompt = f"""你是严谨但鼓励学生的高中{subject}教师。请批改答案。
题目：{question}
学生答案：{student_answer}
请先独立求解再判断，分数为 0 到 100 的整数。反馈要具体，严格输出要求的 JSON 字段。"""
        try:
            result = _invoke_structured(llm_factory, GradeResult, prompt)
            grade = GradeResult.model_validate(result).model_dump()
        except Exception as exc:
            error_type = type(exc).__name__
            return (
                f"答案批改暂时失败，请稍后重试。错误类型：{error_type}",
                {"error_type": error_type},
            )

        verdict = "正确" if grade["correct"] else "需要订正"
        content = (
            f"批改结果：{verdict}（{grade['score']} 分）\n"
            f"反馈：{grade['feedback']}\n"
            f"错误原因：{grade['error_reason'] or '无'}\n"
            f"提示：{grade['hint']}\n"
            f"参考解法：{grade['reference_solution']}"
        )
        return content, {"grade": grade}

    return [
        StructuredTool.from_function(
            func=search_textbook,
            name="search_textbook",
            description=(
                f"检索当前{subject}教材。知识、概念、公式、定义、解题方法、事实类问题必须使用；"
                "普通寒暄不使用。"
            ),
            args_schema=SearchTextbookInput,
            response_format="content_and_artifact",
        ),
        StructuredTool.from_function(
            func=generate_exercise,
            name="generate_exercise",
            description="学生要求出题、练习、测试或调整难度时必须使用。",
            args_schema=GenerateExerciseInput,
            response_format="content_and_artifact",
        ),
        StructuredTool.from_function(
            func=grade_answer,
            name="grade_answer",
            description="学生提交答案、计算结果、解题过程或要求批改时必须使用。",
            args_schema=GradeAnswerInput,
            response_format="content_and_artifact",
        ),
    ]
