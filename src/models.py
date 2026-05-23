from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DocType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class PageType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    MIXED = "mixed"
    IMAGE_ONLY = "image_only"


@dataclass
class PageBlock:
    page: int
    text: str
    source: str
    doc_type: DocType = DocType.PDF
    has_tables: bool = False
    page_type: PageType = PageType.TEXT

    @property
    def char_count(self) -> int:
        return len(self.text)


ParsedPage = PageBlock


@dataclass
class Chunk:
    text: str
    parent_text: str
    source: str
    page: int
    parent_idx: int
    chunk_idx: int = 0
    is_table: bool = False
    page_type: str = PageType.TEXT.value


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int | str
    page_type: str
    is_table: bool
    score: float


@dataclass
class DocumentStats:
    total_pages: int
    parsed_pages: int
    text_pages: int = 0
    table_pages: int = 0
    mixed_pages: int = 0
    image_only_pages: int = 0
    total_tables: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class IndexStats:
    source: str
    doc_type: str
    pages: int
    chunks: int
    tables_found: int = 0
    skipped_pages: int = 0
    warnings: list[str] = field(default_factory=list)