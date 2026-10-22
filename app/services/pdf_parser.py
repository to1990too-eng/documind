"""Извлечение текста из PDF через pypdf.

Сервис не знает ничего про FastAPI и HTTP — только про байты и текст.
Это упрощает тестирование (можно тестировать без запуска сервера).
"""
import io

from pypdf import PdfReader


class PdfParseError(Exception):
    """Ошибка парсинга PDF: файл повреждён, не PDF или не содержит текста."""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Извлекает весь текст из PDF-файла.

    Args:
        file_bytes: содержимое PDF в виде байтов.

    Returns:
        Объединённый текст всех страниц (страницы разделены \\n).

    Raises:
        PdfParseError: если файл не читается как PDF или текст пуст
                       (например, скан без OCR).
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf кидает разные исключения (PdfReadError и др.)
        raise PdfParseError(f"Не удалось прочитать PDF: {exc}") from exc

    text = "\n".join(pages).strip()
    if not text:
        raise PdfParseError(
            "PDF не содержит извлекаемого текста (возможно, это скан без OCR)."
        )
    return text