"""Локальные эмбеддинги через sentence-transformers.

Оборачиваем модель в интерфейс langchain_core.embeddings.Embeddings,
чтобы langchain_chroma.Chroma мог использовать её напрямую.

Модель качается один раз в ~/.cache/huggingface и переиспользуется.
Нормализуем векторы (normalize_embeddings=True), потому что ChromaDB
по умолчанию использует косинусное сходство — для него важна нормировка.
"""
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.config import get_settings


class SentenceTransformerEmbeddings(Embeddings):
    """Обёртка sentence-transformers под интерфейс LangChain."""

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Векторизует список документов (чанков)."""
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Векторизует один запрос (вопрос пользователя)."""
        vector = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()


@lru_cache
def get_embeddings() -> SentenceTransformerEmbeddings:
    """Кэшированный инстанс эмбеддингов (синглтон).

    Модель загружается в память один раз на процесс.
    В тестах можно сбросить через get_embeddings.cache_clear().
    """
    settings = get_settings()
    return SentenceTransformerEmbeddings(settings.embedding_model_name)