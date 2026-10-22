"""Нарезка текста на перекрывающиеся чанки через LangChain.

RecursiveCharacterTextSplitter пытается резать по естественным границам:
сначала по абзацам (\\n\\n), потом по строкам, потом по предложениям.
Это даёт более осмысленные чанки, чем просто резка по N символов.
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


def split_text(
    text: str,
    *,
    metadata: dict[str, str] | None = None,
) -> list[Document]:
    """Режет текст на чанки с перекрытием.

    Args:
        text: исходный текст документа.
        metadata: метаданные, которые прокинутся в каждый чанк
                  (document_id, filename). Важно для фильтрации при поиске.

    Returns:
        Список Document с page_content и metadata.
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    base_meta = metadata or {}
    return [Document(page_content=chunk, metadata=dict(base_meta)) for chunk in chunks]