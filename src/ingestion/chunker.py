from __future__ import annotations

import re
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import Chunk, PageBlock as ParsedPage, PageType

logger = logging.getLogger(__name__)

_MIN_CHILD_LEN = 50
_TABLE_RE = re.compile(
    r"(?:(?:\|[^\n]+\|\n)+\|[-| :]+\|\n(?:\|[^\n]+\|\n)*)",
    re.MULTILINE,
)


class Chunker:
    def __init__(self, parent_chunk_size=1500, child_chunk_size=400, chunk_overlap=50):
        self._parent_spl = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )
        self._child_spl = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def split(self, pages):
        chunks = []
        parent_idx = 0
        chunk_idx = 0

        for page in pages:
            for block_text, is_table in self._extract_blocks(page):
                block_text = block_text.strip()
                if not block_text:
                    continue

                if is_table:
                    if len(block_text) >= _MIN_CHILD_LEN:
                        chunks.append(Chunk(
                            text=block_text,
                            parent_text=block_text,
                            source=page.source,
                            page=page.page,
                            parent_idx=parent_idx,
                            chunk_idx=chunk_idx,
                            is_table=True,
                            page_type=PageType.TABLE.value,
                        ))
                        chunk_idx += 1
                    parent_idx += 1
                else:
                    for parent_text in self._parent_spl.split_text(block_text):
                        for child_text in self._child_spl.split_text(parent_text):
                            if len(child_text.strip()) < _MIN_CHILD_LEN:
                                continue
                            chunks.append(Chunk(
                                text=child_text,
                                parent_text=parent_text,
                                source=page.source,
                                page=page.page,
                                parent_idx=parent_idx,
                                chunk_idx=chunk_idx,
                                is_table=False,
                                page_type=page.page_type.value,
                            ))
                            chunk_idx += 1
                        parent_idx += 1

        logger.info(f"Chunker: {len(chunks)} чанков из {len(pages)} страниц")
        return chunks

    def _extract_blocks(self, page):
        if not page.has_tables:
            return [(page.text, False)]

        blocks = []
        last_end = 0
        for match in _TABLE_RE.finditer(page.text):
            before = page.text[last_end:match.start()].strip()
            if before:
                blocks.append((before, False))
            table = match.group(0).strip()
            if table:
                blocks.append((table, True))
            last_end = match.end()

        after = page.text[last_end:].strip()
        if after:
            blocks.append((after, False))

        return blocks or [(page.text, False)]