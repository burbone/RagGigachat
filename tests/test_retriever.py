import pytest
from unittest.mock import MagicMock
from src.retrieval.retriever import HybridRetriever


def _make_vector_store(chunks):
    vs = MagicMock()
    vs.search.return_value = [
        {
            "text": c.text,
            "parent_text": c.parent_text,
            "source": c.source,
            "page": c.page,
            "page_type": c.page_type,
            "is_table": c.is_table,
            "distance": 0.1,
        }
        for c in chunks
    ]
    return vs


def test_retrieve_returns_results(sample_chunks, mock_llm):
    vs = _make_vector_store(sample_chunks)
    retriever = HybridRetriever(vs, sample_chunks, llm_client=mock_llm)
    results = retriever.retrieve("какая комиссия", top_k=2)
    assert len(results) > 0


def test_table_bonus_for_numeric_query(sample_chunks, mock_llm):
    vs = _make_vector_store(sample_chunks)
    retriever = HybridRetriever(vs, sample_chunks, llm_client=mock_llm)
    results = retriever.retrieve("лимит 50000 рублей", top_k=2)
    has_table = any(r.is_table for r in results)
    assert has_table


def test_no_duplicate_parents(sample_chunks, mock_llm):
    vs = _make_vector_store(sample_chunks)
    retriever = HybridRetriever(vs, sample_chunks, llm_client=mock_llm)
    results = retriever.retrieve("карта", top_k=5)
    parent_texts = [r.text for r in results]
    assert len(parent_texts) == len(set(parent_texts))


def test_retrieve_without_llm(sample_chunks):
    vs = _make_vector_store(sample_chunks)
    retriever = HybridRetriever(vs, sample_chunks, llm_client=None)
    results = retriever.retrieve("вопрос", top_k=2)
    assert isinstance(results, list)