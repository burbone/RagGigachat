from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

_TABLE_TRIGGER = re.compile(
    r"\d|сколько|стоимость|тариф|комиссия|процент|лимит|сумма|цена|размер|плата|обслуживание|перевыпуск",
    re.IGNORECASE,
)
_TABLE_BONUS = 0.35
_RRF_K = 30


@dataclass
class _Candidate:
    child_text: str
    parent_text: str
    source: str
    page: int
    is_table: bool
    page_type: str = "text"
    rrf_score: float = 0.0


class HybridRetriever:
    def __init__(self, vector_store, chunks, llm_client=None, reranker_model=None):
        self._vs = vector_store
        self._chunks = chunks
        self._llm = llm_client
        self._reranker = None
        self._reranker_model = reranker_model or os.getenv(
            "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        )
        self._use_reranker = os.getenv("USE_RERANKER", "true").lower() == "true"

        tokenized = [c.text.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

        logger.info(f"HybridRetriever: {len(chunks)} чанков reranker={self._use_reranker}")

    def _get_reranker(self):
        if self._reranker is None and self._use_reranker:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self._reranker_model)
            logger.info(f"Reranker загружен: {self._reranker_model}")
        return self._reranker

    def retrieve(self, query, top_k=4):
        queries = self._expand_query(query)
        pool_size = top_k * 4 if self._use_reranker else top_k * 2
        candidates = self._rrf_candidates(queries, top_k=pool_size)

        if not candidates:
            logger.warning(f"Retrieval: 0 кандидатов для '{query}'")
            return []

        reranker = self._get_reranker()
        needs_table = bool(_TABLE_TRIGGER.search(query))

        if reranker:
            pairs = [[query, c.child_text] for c in candidates]
            scores = reranker.predict(pairs).tolist()
            for score, cand in zip(scores, candidates):
                cand.rrf_score = float(score) + (_TABLE_BONUS if needs_table and cand.is_table else 0.0)
        else:
            for cand in candidates:
                cand.rrf_score += _TABLE_BONUS if needs_table and cand.is_table else 0.0

        candidates.sort(key=lambda c: c.rrf_score, reverse=True)

        seen = set()
        results = []
        for cand in candidates:
            if len(results) >= top_k:
                break
            if cand.parent_text in seen:
                continue
            seen.add(cand.parent_text)
            results.append(RetrievedChunk(
                text=cand.parent_text,
                source=cand.source,
                page=cand.page,
                page_type=cand.page_type,
                is_table=cand.is_table,
                score=cand.rrf_score,
            ))

        return results

    def _rrf_candidates(self, queries, top_k):
        cand_map = {}

        for q in queries:
            for rank, hit in enumerate(self._vs.search_raw_child(q, top_k=top_k)):
                key = hit["child_text"]
                if key not in cand_map:
                    cand_map[key] = _Candidate(
                        child_text=hit["child_text"],
                        parent_text=hit["parent_text"],
                        source=hit["source"],
                        page=hit["page"],
                        is_table=hit.get("is_table", False),
                        page_type=hit.get("page_type", "text"),
                    )
                cand_map[key].rrf_score += 1.0 / (_RRF_K + rank + 1)

            for rank, hit in enumerate(self._bm25_search(q, top_k=top_k)):
                key = hit["child_text"]
                if key not in cand_map:
                    cand_map[key] = _Candidate(
                        child_text=hit["child_text"],
                        parent_text=hit["parent_text"],
                        source=hit["source"],
                        page=hit["page"],
                        is_table=hit.get("is_table", False),
                        page_type=hit.get("page_type", "text"),
                    )
                cand_map[key].rrf_score += 1.0 / (_RRF_K + rank + 1)

        return sorted(cand_map.values(), key=lambda c: c.rrf_score, reverse=True)[:top_k]

    def _bm25_search(self, query, top_k):
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        max_s = max(scores) if scores.any() else 1.0

        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in top_idx:
            if scores[i] <= 0:
                break
            c = self._chunks[i]
            results.append({
                "child_text": c.text,
                "parent_text": c.parent_text,
                "source": c.source,
                "page": c.page,
                "is_table": getattr(c, "is_table", False),
                "page_type": getattr(c, "page_type", "text"),
                "score": float(scores[i]) / max_s,
            })
        return results

    def _expand_query(self, query):
        if not self._llm or len(query.strip()) < 10:
            return [query]
        try:
            variants = self._llm.expand_query(query)
            seen = {query}
            out = [query]
            for v in variants:
                if v not in seen and v.strip():
                    out.append(v)
                    seen.add(v)
                if len(out) == 3:
                    break
            return out
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]