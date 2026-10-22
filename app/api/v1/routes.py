"""Эндпоинты API v1."""
import logging
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas import UploadResponse
from app.services.pdf_parser import PdfParseError, extract_text_from_pdf
from app.services.text_splitter import split_text
from app.services.vector_store import add_chunks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["v1"])


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Загружает PDF, парсит, режет на чанки и сохраняет в ChromaDB."""
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ожидается PDF-файл.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пустой файл.",
        )

    try:
        text = extract_text_from_pdf(file_bytes)
    except PdfParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    document_id = str(uuid.uuid4())
    filename = file.filename or "unknown.pdf"
    chunks = split_text(
        text,
        metadata={"document_id": document_id, "filename": filename},
    )

    try:
        add_chunks(chunks)
    except Exception as exc:
        # Векторизация/запись в Chroma упали — это наша проблема, не клиента.
        logger.exception("Не удалось сохранить чанки в ChromaDB")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка сохранения документа в векторную БД.",
        ) from exc

    return UploadResponse(
        document_id=document_id,
        filename=filename,
        chunks_count=len(chunks),
    )