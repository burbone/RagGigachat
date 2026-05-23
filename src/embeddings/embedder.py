from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class LocalEmbedder:
    def __init__(self, model_name=None):
        from sentence_transformers import SentenceTransformer

        name = model_name or os.getenv("LOCAL_EMBEDDING_MODEL", _DEFAULT_MODEL)
        self._model = SentenceTransformer(name)
        self._dim = self._model.get_sentence_embedding_dimension()

        cache_dir = os.getenv("EMBEDDING_CACHE", "")
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Embedder: '{name}' dim={self._dim} cache={bool(self._cache_dir)}")

    def embed_documents(self, texts):
        if self._cache_dir:
            return self._embed_with_disk_cache(texts)
        return self._encode_batch(texts)

    def embed_query(self, text):
        return self._embed_query_lru(text)

    @lru_cache(maxsize=512)
    def _embed_query_lru(self, text):
        vec = self._model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return vec.tolist()

    def _encode_batch(self, texts):
        vecs = self._model.encode(
            texts,
            batch_size=128,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    def _embed_with_disk_cache(self, texts):
        results = [None] * len(texts)
        missing = []

        for i, text in enumerate(texts):
            hit = self._cache_load(text)
            if hit is not None:
                results[i] = hit
            else:
                missing.append(i)

        if missing:
            vecs = self._encode_batch([texts[i] for i in missing])
            for idx, vec in zip(missing, vecs):
                results[idx] = vec
                self._cache_save(texts[idx], vec)

        logger.debug(f"Embedder cache: {len(texts) - len(missing)} hit {len(missing)} miss")
        return results

    def _cache_path(self, text):
        h = hashlib.sha256(text.encode()).hexdigest()
        return self._cache_dir / f"{h}.npy"

    def _cache_load(self, text):
        p = self._cache_path(text)
        return np.load(p).tolist() if p.exists() else None

    def _cache_save(self, text, vec):
        np.save(self._cache_path(text), np.array(vec, dtype=np.float32))


HybridEmbedder = LocalEmbedder