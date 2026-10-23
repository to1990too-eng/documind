"""RAG-пайплайн: поиск релевантных чанков + генерация ответа.

Сервис не знает про FastAPI — принимает строки, возвращает структуру.
Это позволяет тестировать его отдельно от HTTP-слоя.
"""
from dataclasses import dataclass

from app.services.llm_service import LLMService
from app.services.vector_store import search_chunks

SYSTEM_PROMPT = (
    "Ты — помощник по документам. Отвечай на вопрос пользователя, "
    "используя ТОЛЬКО предоставленный контекст. "
    "Если в контексте нет ответа — честно скажи: "
    "«В документе нет информации об этом». "
    "Не выдумывай факты. Отвечай кратко и по делу, на русском языке."
)


@dataclass(frozen=True)
class RagResult:
    """Результат RAG-пайплайна."""
    answer: str
    sources: list[str]


def _build_user_prompt(question: str, context_chunks: list[str]) -> str:
    """Склеивает контекст и вопрос в user-промпт."""
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(контекст пуст)"
    return (
        f"Контекст из документа:\n\n{context}\n\n"
        f"Вопрос пользователя: {question}"
    )


def answer_question(
    *,
    document_id: str,
    question: str,
    llm: LLMService,
) -> RagResult:
    """Полный RAG-цикл: поиск → промпт → генерация → ответ.

    Args:
        document_id: UUID документа (фильтр поиска).
        question: вопрос пользователя.
        llm: реализация LLMService (DummyLLM или LocalLLM).

    Returns:
        RagResult с ответом и списком источников (тексты чанков).
    """
    docs = search_chunks(document_id, question)
    context_chunks = [d.page_content for d in docs]

    if not context_chunks:
        return RagResult(
            answer="В документе нет информации об этом (не найдено релевантных фрагментов).",
            sources=[],
        )

    user_prompt = _build_user_prompt(question, context_chunks)
    answer = llm.generate(SYSTEM_PROMPT, user_prompt)
    return RagResult(answer=answer, sources=context_chunks)