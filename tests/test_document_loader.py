import pytest
from src.ingestion.document_loader import DocumentLoader, PageType


loader = DocumentLoader()


def test_classify_text_page():
    from src.ingestion.document_loader import ParsedPage
    page = loader._classify_and_build(
        text="Это обычный текстовый абзац без таблиц. " * 5,
        page_num=1,
        source="test.pdf",
        extra_tables=[],
    )
    assert page.page_type == PageType.TEXT
    assert not page.has_tables
    assert not page.warnings


def test_classify_image_only():
    page = loader._classify_and_build(
        text="",
        page_num=5,
        source="test.pdf",
        extra_tables=[],
    )
    assert page.page_type == PageType.IMAGE_ONLY
    assert len(page.warnings) > 0
    assert "5" in page.warnings[0]


def test_classify_table_page():
    table_md = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    page = loader._classify_and_build(
        text="Небольшой заголовок",
        page_num=3,
        source="test.pdf",
        extra_tables=[table_md],
    )
    assert page.has_tables
    assert page.page_type in (PageType.TABLE, PageType.MIXED)


def test_table_to_markdown():
    raw = [["Имя", "Сумма"], ["Иван", "1000"], ["Пётр", None]]
    md = loader._table_to_markdown(raw)
    assert "Имя" in md
    assert "1000" in md
    assert "---" in md
    assert md.count("|") > 0


def test_has_markdown_table():
    text_with = "Привет\n| A | B |\n|---|---|\n| 1 | 2 |"
    text_without = "Обычный текст без таблицы"
    assert loader._has_markdown_table(text_with)
    assert not loader._has_markdown_table(text_without)