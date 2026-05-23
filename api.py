from __future__ import annotations

import asyncio
import logging
import shutil
import logging.handlers
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            "data/app.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__name__)


class _State:
    def __init__(self):
        from src.pipeline import RAGPipeline
        self.pipeline = RAGPipeline()
        self.lock = asyncio.Lock()


_state: _State | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state
    Path("data").mkdir(exist_ok=True)
    _state = _State()
    logger.info("Pipeline инициализирован")
    yield


app = FastAPI(title="RAG Ассистент", version="3.0", lifespan=lifespan)


class UploadResponse(BaseModel):
    status: str
    filename: str
    pages: int
    chunks: int
    tables_found: int
    skipped_pages: int
    warnings: list[str]


class AskRequest(BaseModel):
    question: str


class SourceRef(BaseModel):
    page: int
    source: str
    is_table: bool
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


class StatusResponse(BaseModel):
    indexed: bool
    filename: str | None
    pages: int
    chunks: int
    tables_found: int
    skipped_pages: int


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".docx", ".txt", ".md")):
        raise HTTPException(400, "Поддерживаются: PDF, DOCX, TXT, MD")

    async with _state.lock:
        _state.pipeline.reset()

        upload_dir = Path("data/documents")
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / file.filename

        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            result = await _state.pipeline.index(str(save_path))
        except Exception as e:
            logger.exception("Ошибка индексирования")
            raise HTTPException(500, str(e))

    warnings = _state.pipeline.doc_stats.warnings if _state.pipeline.doc_stats else []

    return UploadResponse(
        status="ok",
        filename=file.filename,
        pages=result.pages,
        chunks=result.chunks,
        tables_found=result.tables_found,
        skipped_pages=result.skipped_pages,
        warnings=warnings,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not _state.pipeline.is_indexed:
        raise HTTPException(400, "Сначала загрузите документ")
    if not req.question.strip():
        raise HTTPException(400, "Вопрос не может быть пустым")

    try:
        answer, sources = await _state.pipeline.ask(req.question)
    except Exception as e:
        logger.exception("Ошибка генерации")
        raise HTTPException(500, str(e))

    return AskResponse(
        answer=answer,
        sources=[
            SourceRef(
                page=s.page,
                source=s.source,
                is_table=s.is_table,
                score=round(s.score, 3),
            )
            for s in sources
        ],
    )


@app.get("/status", response_model=StatusResponse)
async def status():
    stats = _state.pipeline.doc_stats
    return StatusResponse(
        indexed=_state.pipeline.is_indexed,
        filename=stats.warnings[0].split(":")[0] if stats and stats.warnings else None,
        pages=stats.parsed_pages if stats else 0,
        chunks=len(_state.pipeline._chunks),
        tables_found=stats.total_tables if stats else 0,
        skipped_pages=stats.image_only_pages if stats else 0,
    )


#для отладки
@app.post("/debug/retrieve")
async def debug_retrieve(req: AskRequest):
    if not _state.pipeline.is_indexed:
        raise HTTPException(400, "Сначала загрузите документ")
    from src.retrieval.retriever import HybridRetriever
    retriever = HybridRetriever(
        vector_store=_state.pipeline._vs,
        chunks=_state.pipeline._chunks,
        llm_client=_state.pipeline._llm,
    )
    relevant = retriever.retrieve(req.question, top_k=4)
    return {
        "question": req.question,
        "chunks": [
            {
                "page": c.page,
                "is_table": c.is_table,
                "score": round(c.score, 3),
                "text_preview": c.text[:500],
            }
            for c in relevant
        ],
    }


app.mount("/static", StaticFiles(directory="static"), name="static")