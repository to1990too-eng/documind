"""Работа с ChromaDB: сохранение чанков и поиск по ним.

Используем PersistentClient — данные сохраняются в директории
CHROMA_PERSIST_DIR и переживают перезапуск приложения.

Коллекция одна на всё приложение ("documind"), но поиск всегда
фильтруется по metadata["document_id"], чтобы пользователь
не получал чанки из чужих документов.
"""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import get_settings
from app.services.embeddings import get_embeddings

COLLECTION_NAME = "documind"


@lru_cache
def get_vector_store() -> Chroma:
    """Кэшированный инстанс Chroma (синглтон на процесс).

    При создании Chroma с persist_directory + collection_name
    автоматически подключается к существующей коллекции или создаёт новую.
    """
    settings = get_settings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
        collection_metadata={"hnsw:space": "cosine"},  # косинусное сходство
    )


def add_chunks(chunks: list[Document]) -> list[str]:
    """Сохраняет чанки в ChromaDB.

    Args:
        chunks: список Document (page_content + metadata с document_id).

    Returns:
        Список внутренних ID, присвоенных Chroma.
    """
    if not chunks:
        return []
    store = get_vector_store()
    return store.add_documents(chunks)


def search_chunks(
    document_id: str,
    query: str,
    *,
    k: int | None = None,
) -> list[Document]:
    """Ищет top-k релевантных чанков внутри одного документа.

    Args:
        document_id: UUID документа — фильтр по metadata.
        query: текст вопроса.
        k: сколько чанков вернуть (по умолчанию из настроек).

    Returns:
        Список Document, отсортированный по релевантности (лучший первым).
    """
    settings = get_settings()
    store = get_vector_store()
    return store.similarity_search(
        query,
        k=k or settings.retrieval_top_k,
        filter={"document_id": document_id},
    )