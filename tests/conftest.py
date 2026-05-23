import pytest
from unittest.mock import MagicMock

from src.ingestion.document_loader import ParsedPage, PageType
from src.ingestion.chunker import Chunk


@pytest.fixture
def sample_text_page():
    return ParsedPage(
        page=1,
        text="Дебетовая карта позволяет оплачивать покупки. Комиссия за снятие наличных составляет 1.5%. "
             "Максимальный лимит в день - 100 000 рублей. Карта действительна 3 года.",
        page_type=PageType.TEXT,
        source="test.pdf",
        char_count=200,
    )


@pytest.fixture
def sample_table_page():
    return ParsedPage(
        page=2,
        text="| Тариф | Стоимость | Лимит |\n|---|---|---|\n| Базовый | 0 руб. | 50 000 |\n| Премиум | 299 руб. | 500 000 |",
        page_type=PageType.TABLE,
        source="test.pdf",
        has_tables=True,
        table_count=1,
        char_count=120,
    )


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            text="Комиссия за снятие наличных составляет 1.5%.",
            parent_text="Дебетовая карта позволяет оплачивать покупки. Комиссия за снятие наличных составляет 1.5%.",
            source="test.pdf",
            page=1,
            page_type="text",
            parent_idx=0,
            is_table=False,
        ),
        Chunk(
            text="| Тариф | Стоимость |\n|---|---|\n| Базовый | 0 руб. |",
            parent_text="| Тариф | Стоимость |\n|---|---|\n| Базовый | 0 руб. |",
            source="test.pdf",
            page=2,
            page_type="table",
            parent_idx=1,
            is_table=True,
        ),
    ]


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_documents.side_effect = lambda texts: [[0.1] * 128 for _ in texts]
    embedder.embed_query.return_value = [0.1] * 128
    return embedder


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.expand_query.return_value = ["вопрос", "альтернатива 1"]
    llm.generate.return_value = "Тестовый ответ [стр. 1]"
    return llm