import pytest
from src.ingestion.chunker import Chunker
from src.ingestion.document_loader import ParsedPage, PageType


def test_text_page_creates_chunks(sample_text_page):
    chunker = Chunker(parent_chunk_size=200, child_chunk_size=80, chunk_overlap=10)
    chunks = chunker.split([sample_text_page])
    assert len(chunks) > 0
    for c in chunks:
        assert c.source == "test.pdf"
        assert c.page == 1
        assert len(c.text) >= 50


def test_table_not_split(sample_table_page):
    chunker = Chunker(parent_chunk_size=200, child_chunk_size=80, chunk_overlap=10)
    chunks = chunker.split([sample_table_page])
    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) == 1
    assert "Базовый" in table_chunks[0].text
    assert "Премиум" in table_chunks[0].text


def test_parent_child_relationship(sample_text_page):
    chunker = Chunker(parent_chunk_size=200, child_chunk_size=80, chunk_overlap=10)
    chunks = chunker.split([sample_text_page])
    for c in chunks:
        if not c.is_table:
            assert len(c.parent_text) >= len(c.text)


def test_min_chunk_length(sample_text_page):
    chunker = Chunker(parent_chunk_size=200, child_chunk_size=80, chunk_overlap=10)
    chunks = chunker.split([sample_text_page])
    for c in chunks:
        assert len(c.text.strip()) >= 50


def test_multiple_pages(sample_text_page, sample_table_page):
    chunker = Chunker(parent_chunk_size=300, child_chunk_size=100, chunk_overlap=10)
    chunks = chunker.split([sample_text_page, sample_table_page])
    pages = {c.page for c in chunks}
    assert 1 in pages
    assert 2 in pages