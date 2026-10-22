"""Точка входа FastAPI-приложения DocuMind."""
from fastapi import FastAPI

from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="DocuMind",
    description="RAG-система: вопросы по PDF-документам (локальные эмбеддинги)",
    version="0.1.0",
)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Health-check. Показывает, какой провайдер LLM выбран (на Шаге 4)."""
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model_name,
    }