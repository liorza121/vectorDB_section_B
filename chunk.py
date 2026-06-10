"""Preprocessing and chunking utilizing production text splitters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import entry_text

@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str

# Initialize the optimized splitter globally so it isn't rebuilt on every record
_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=175,
    chunk_overlap=40
)

def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Split one corpus entry using LangChain's token-aware recursive character splitter.
    """
    page_id = int(record["page_id"])
    title = record.get("title", "").strip()
    text = entry_text(record)

    if not text.strip():
        full_text = f"{title}:" if title else ""
        return [Chunk(page_id=page_id, chunk_id=0, text=full_text)]

    # 1. Use the ready-made method to instantly slice the text cleanly
    text_segments = _SPLITTER.split_text(text)

    chunks: List[Chunk] = []
    for chunk_id, segment in enumerate(text_segments):
        # 2. Prepend the page title to preserve global context
        chunk_text = f"{title}: {segment}" if title else segment

        # 3. Track seamlessly back to parent page_id
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=chunk_text))

    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks