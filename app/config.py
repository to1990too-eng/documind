"""Настройки приложения через Pydantic Settings.

Читаются из переменных окружения / .env.
Все значения имеют дефолты, чтобы приложение запускалось без .env.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Эмбеддинги (всегда локальные)
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # LLM (Шаг 4; пока dummy = возвращает заглушку)
    llm_provider: Literal["dummy", "local"] = "dummy"

    # RAG
    chroma_persist_dir: str = "./data/chroma"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    """Кэшированный инстанс настроек (синглтон).

    В тестах можно сбросить через get_settings.cache_clear().
    """
    return Settings()