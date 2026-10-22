"""Pydantic-схемы запросов и ответов API."""
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Ответ после загрузки PDF."""
    document_id: str = Field(..., description="UUID документа")
    filename: str
    chunks_count: int = Field(..., ge=0)


class AskRequest(BaseModel):
    """Запрос к /chat/ask (Шаг 4)."""
    document_id: str
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    """Ответ от /chat/ask (Шаг 4)."""
    answer: str
    sources: list[str] = Field(default_factory=list)