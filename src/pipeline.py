from __future__ import annotations

import asyncio
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.ingestion.document_loader import DocumentLoader, DocumentStats
from src.ingestion.chunker import Chunker
from src.embeddings.embedder import HybridEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import HybridRetriever
from src.generation.llm import GigaChatLLM
from src.models import Chunk, IndexStats, RetrievedChunk

logger = logging.getLogger(__name__)

_STATE_FILE = Path(os.getenv("STATE_PATH", "./data/pipeline_state.pkl"))
_EXECUTOR = ThreadPoolExecutor(max_workers=2)


class RAGPipeline:
    def __init__(self):
        self._loader = DocumentLoader()
        self._chunker = Chunker(
            parent_chunk_size=int(os.getenv("PARENT_CHUNK_SIZE", 1500)),
            child_chunk_size=int(os.getenv("CHILD_CHUNK_SIZE", 400)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50)),
        )
        self._embedder = HybridEmbedder()
        self._vs = VectorStore(self._embedder)
        self._llm = GigaChatLLM()
        self._top_k = int(os.getenv("RETRIEVAL_TOP_K", 6))

        self._chunks = []
        self._indexed = False
        self._doc_stats = None
        self._retriever = None

        self._try_restore_state()

    def _try_restore_state(self):
        if not _STATE_FILE.exists():
            return
        try:
            with open(_STATE_FILE, "rb") as f:
                state = pickle.load(f)
            self._chunks = state["chunks"]
            self._doc_stats = state["doc_stats"]
            self._indexed = True
            self._retriever = HybridRetriever(
                vector_store=self._vs,
                chunks=self._chunks,
                llm_client=self._llm,
            )
            logger.info(f"Pipeline: восстановлено состояние ({len(self._chunks)} чанков)")
        except Exception as e:
            logger.warning(f"Pipeline: не удалось восстановить состояние: {e}")

    def _save_state(self):
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_STATE_FILE, "wb") as f:
            pickle.dump({"chunks": self._chunks, "doc_stats": self._doc_stats}, f)

    def _index_sync(self, file_path):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        logger.info(f"=== Индексирую: {path.name} ===")
        pages, stats = self._loader.load(str(path))
        self._doc_stats = stats

        self._chunks = self._chunker.split(pages)
        self._vs.add_chunks(self._chunks)
        self._indexed = True
        self._retriever = HybridRetriever(
            vector_store=self._vs,
            chunks=self._chunks,
            llm_client=self._llm,
        )
        self._save_state()

        result = IndexStats(
            source=path.name,
            doc_type="pdf",
            pages=stats.parsed_pages,
            chunks=len(self._chunks),
            tables_found=stats.total_tables,
            skipped_pages=stats.image_only_pages,
        )
        logger.info(
            f"Готово: страниц={result.pages} чанков={result.chunks} таблиц={result.tables_found} пропущено={result.skipped_pages}"
        )
        return result

    async def index(self, file_path):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_EXECUTOR, self._index_sync, file_path)

    def _ask_sync(self, question):
        if not self._indexed or not self._retriever:
            raise RuntimeError("Сначала вызовите index()")
        relevant = self._retriever.retrieve(question, top_k=self._top_k)
        if not relevant:
            return "Релевантная информация в документе не найдена.", []
        answer = self._llm.generate(question, relevant)
        return answer, relevant

    async def ask(self, question):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_EXECUTOR, self._ask_sync, question)

    def reset(self):
        self._vs.reset()
        self._chunks = []
        self._indexed = False
        self._doc_stats = None
        self._retriever = None
        if _STATE_FILE.exists():
            _STATE_FILE.unlink()
        logger.info("Pipeline сброшен")

    @property
    def is_indexed(self):
        return self._indexed

    @property
    def doc_stats(self):
        return self._doc_stats