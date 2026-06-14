"""Preprocessing and chunking utilizing compliant native text splitters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from utils import entry_text

@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Split one corpus entry using a legal sliding-window word splitter.
    """
    page_id = int(record["page_id"])
    title = record.get("title", "").strip()
    text = entry_text(record)

    if not text.strip():
        full_text = f"{title}:" if title else ""
        return [Chunk(page_id=page_id, chunk_id=0, text=full_text)]

    # Split into words (whitespace tokenization approximation)
    words = text.split()

    # Hyperparameters mapping to your targets
    chunk_size = 175
    chunk_overlap = 40
    step = chunk_size - chunk_overlap

    chunks: List[Chunk] = []
    chunk_id = 0

    # Handle tiny texts that fit in one block
    if len(words) <= chunk_size:
        segment = " ".join(words)
        chunk_text = f"{title}: {segment}" if title else segment
        return [Chunk(page_id=page_id, chunk_id=0, text=chunk_text)]

    # Sliding window loop
    for i in range(0, len(words), step):
        window = words[i : i + chunk_size]
        if not window:
            break

        segment = " ".join(window)
        chunk_text = f"{title}: {segment}" if title else segment
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=chunk_text))
        chunk_id += 1

        # Stop if we processed the end of the text
        if i + chunk_size >= len(words):
            break

    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks