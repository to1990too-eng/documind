"""LLM-сервисы: интерфейс + две реализации (dummy и local).

Protocol LLMService позволяет подменять реализацию без изменения
вызывающего кода (rag_service). В тестах используется DummyLLM,
в проде можно включить LocalLLM через LLM_PROVIDER=local.
"""
import logging
from functools import lru_cache
from typing import Protocol

from app.config import get_settings

logger = logging.getLogger(__name__)


class LLMService(Protocol):
    """Интерфейс LLM. Реализации: DummyLLM, LocalLLM."""

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class DummyLLM:
    """Заглушка без моделей. Возвращает extractive-ответ.

    Не «генерирует» текст, а аккуратно склеивает найденные фрагменты.
    Этого достаточно, чтобы протестировать весь RAG-пайплайн
    (поиск, промпт, источники) без скачивания моделей.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # user_prompt содержит контекст + вопрос; возвращаем его начало.
        # Это не претендует на качество — только на работоспособность пайплайна.
        snippet = user_prompt.strip().split("\n\n")[0]
        return (
            "[DUMMY LLM] Включите LLM_PROVIDER=local для генерации. "
            f"Найденный контекст:\n\n{snippet}"
        )


class LocalLLM:
    """Локальная LLM через transformers.pipeline.

    Модель качается один раз в ~/.cache/huggingface.
    Работает на CPU (медленно, но без GPU). Для портфолио — достаточно.
    """

    def __init__(self, model_name: str, max_new_tokens: int, device: str) -> None:
        # Импорт внутри __init__, чтобы не тянуть transformers,
        # если используется DummyLLM.
        from transformers import pipeline

        logger.info("Загружаю локальную LLM: %s (device=%s)", model_name, device)
        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            torch_dtype="auto",
        )
        self._max_new_tokens = max_new_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Chat-шаблон Qwen2.5: system + user.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = self._pipe(
            messages,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,          # детерминированно
            return_full_text=False,
        )
        # pipeline возвращает list[dict] с ключом generated_text
        return result[0]["generated_text"].strip()


@lru_cache
def get_llm_service() -> LLMService:
    """Фабрика LLM-сервиса по настройкам.

    Кэшируется: LocalLLM загружается в память один раз на процесс.
    В тестах можно подменить через dependency_overrides или cache_clear.
    """
    settings = get_settings()
    if settings.llm_provider == "local":
        return LocalLLM(
            model_name=settings.local_llm_model,
            max_new_tokens=settings.local_llm_max_new_tokens,
            device=settings.local_llm_device,
        )
    return DummyLLM()