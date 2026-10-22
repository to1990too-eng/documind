# DocuMind

RAG-система для вопросов по загруженным PDF-документам.

**Стек:** FastAPI, LangChain, ChromaDB, sentence-transformers (локальные эмбеддинги), pypdf.

## Запуск

```bash
pip install -e ".[dev]"
uvicorn app.main:app --reload