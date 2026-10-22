"""Эндпоинты API v1.

Пока только загрузка. На Шаге 3 добавим сохранение в ChromaDB,
на Шаге 4 — /chat/ask.
"""
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas import UploadResponse
from app.services.pdf_parser import PdfParseError, extract_text_from_pdf
from app.services.text_splitter import split_text

router = APIRouter(prefix="/api/v1", tags=["v1"])


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Загружает PDF, парсит и режет на чанки.

    На этом шаге чанки НЕ сохраняются в ChromaDB — это Шаг 3.
    Но мы возвращаем их количество, чтобы проверить пайплайн.
    """
    # 1. Проверка типа файла
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ожидается PDF-файл.",
        )

    # 2. Чтение байтов
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пустой файл.",
        )

    # 3. Парсинг
    try:
        text = extract_text_from_pdf(file_bytes)
    except PdfParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # 4. Нарезка
    document_id = str(uuid.uuid4())
    chunks = split_text(
        text,
        metadata={
            "document_id": document_id,
            "filename": file.filename or "unknown.pdf",
        },
    )

    # TODO (Шаг 3): сохранить chunks в ChromaDB.
    return UploadResponse(
        document_id=document_id,
        filename=file.filename or "unknown.pdf",
        chunks_count=len(chunks),
    )