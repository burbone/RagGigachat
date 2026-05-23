# RAG Ассистент

FastAPI + GigaChat + гибридный поиск (BM25 + dense) + CrossEncoder reranking.

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Создать .env
cp .env .env
# Вписать GIGACHAT_CREDENTIALS в .env

# 3. Запустить
uvicorn api:app --reload
# Открыть http://127.0.0.1:8000
```

## Модели эмбеддингов

По умолчанию используется `paraphrase-multilingual-MiniLM-L12-v2` (~120 MB).  
Скачается автоматически при первом запуске.

Если нужно лучшее качество и есть свободные 2.3 GB — в `.env`:
```
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3
```

### Очистить кэш BGE-M3 (если скачалась случайно)

```
# Windows
rmdir /s /q %USERPROFILE%\.cache\huggingface\hub\models--BAAI--bge-m3

# Linux / Mac
rm -rf ~/.cache/huggingface/hub/models--BAAI--bge-m3
```

## CLI

```bash
python main.py index data/documents/file.pdf
python main.py ask   data/documents/file.pdf "Какая комиссия за снятие?"
python main.py serve --reload
```

## Тесты

```bash
pytest tests/ -v
```

## Структура

```
src/
  ingestion/
    document_loader.py   # PDF + DOCX + TXT/MD, таблицы через pdfplumber
    chunker.py           # Parent-Child чанкинг, таблицы не разрезаются
  embeddings/
    embedder.py          # GigaChat API + локальная модель, гибрид
  retrieval/
    vector_store.py      # ChromaDB
    retriever.py         # BM25 + dense + query expansion + CrossEncoder rerank
  generation/
    llm.py               # GigaChat LLM
  pipeline.py            # Оркестратор
  models.py              # Все типы данных
api.py                   # FastAPI
main.py                  # CLI (typer)
```