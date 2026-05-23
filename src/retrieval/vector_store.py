from __future__ import annotations

import logging
import os

import chromadb
from chromadb.config import Settings

from src.models import Chunk, RetrievedChunk
from src.embeddings.embedder import HybridEmbedder

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "documents"
_BATCH_SIZE = 500


class VectorStore:
    def __init__(self, embedder):
        self._embedder = embedder
        chroma_path = os.getenv("CHROMA_PATH", "./data/chroma")
        self._client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"VectorStore: коллекция '{_COLLECTION_NAME}' готова")

    def add_chunks(self, chunks):
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_documents(texts)

        ids = [f"{c.source}_p{c.page}_{c.chunk_idx}" for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "page": c.page,
                "parent_text": c.parent_text,
                "parent_idx": c.parent_idx,
                "is_table": int(c.is_table),
                "page_type": c.page_type,
            }
            for c in chunks
        ]

        for i in range(0, len(chunks), _BATCH_SIZE):
            self._collection.add(
                ids=ids[i:i+_BATCH_SIZE],
                embeddings=embeddings[i:i+_BATCH_SIZE],
                documents=texts[i:i+_BATCH_SIZE],
                metadatas=metadatas[i:i+_BATCH_SIZE],
            )
        logger.info(f"VectorStore: добавлено {len(chunks)} чанков")

    def search(self, query, top_k=10):
        q_vec = self._embedder.embed_query(query)
        n = min(top_k, self._collection.count() or 1)
        results = self._collection.query(query_embeddings=[q_vec], n_results=n)
        out = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            out.append(RetrievedChunk(
                text=meta.get("parent_text", doc),
                source=meta.get("source", "?"),
                page=int(meta.get("page", 0)),
                page_type=meta.get("page_type", "text"),
                is_table=bool(meta.get("is_table", 0)),
                score=round(1.0 - float(dist), 4),
            ))
        return out

    def search_raw_child(self, query, top_k=10):
        q_vec = self._embedder.embed_query(query)
        n = min(top_k, self._collection.count() or 1)
        results = self._collection.query(query_embeddings=[q_vec], n_results=n)
        out = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            out.append({
                "child_text": doc,
                "parent_text": meta.get("parent_text", doc),
                "source": meta.get("source", "?"),
                "page": int(meta.get("page", 0)),
                "page_type": meta.get("page_type", "text"),
                "is_table": bool(meta.get("is_table", 0)),
                "score": round(1.0 - float(dist), 4),
            })
        return out

    def reset(self):
        try:
            self._client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("VectorStore: коллекция сброшена")

    @property
    def count(self):
        return self._collection.count()