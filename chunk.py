"""Preprocessing and chunking - Sentence Aware & 100% Freeze Proof."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List
from utils import entry_text

@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def get_sentences(text: str) -> List[str]:
    """Pragmatic, linear, and 100% FREEZE-PROOF sentence splitter."""
    clean_text = re.sub(r'\s+', ' ', text).strip()
    if not clean_text:
        return []

    words = clean_text.split()
    sentences = []
    current_sentence = []

    # Common abbreviations to protect from false splits
    abbrev = {
        "dr.", "st.", "mr.", "mrs.", "ms.", "co.", "vs.", "e.g.", "i.e.",
        "vol.", "ed.", "no.", "u.s.", "u.s.a.", "prof.", "inc.", "ltd."
    }

    for word in words:
        current_sentence.append(word)
        word_lower = word.lower()

        if word_lower.endswith(('.', '!', '?')):
            if word_lower in abbrev:
                continue
            if len(word_lower) == 2 and word_lower[0].isalpha():  # e.g., "A."
                continue

            sentences.append(" ".join(current_sentence))
            current_sentence = []

    if current_sentence:
        sentences.append(" ".join(current_sentence))

    return [s for s in sentences if s.strip()]


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Sentence-aware chunking using a safe, guaranteed finite loop.
    Ensures chunks never split in the middle of a sentence.
    """
    page_id = int(record["page_id"])
    title = record.get("title", "").strip()
    text = entry_text(record)

    if not text.strip():
        full_text = f"{title}:" if title else ""
        return [Chunk(page_id=page_id, chunk_id=0, text=full_text)]

    sentences = get_sentences(text)

    target_chunk_words = 250
    target_overlap_words = 80

    chunks: List[Chunk] = []
    chunk_id = 0

    current_chunk = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # If adding this sentence exceeds the limit, seal the current chunk
        if current_words + sentence_words > target_chunk_words and current_chunk:
            segment = " ".join(current_chunk)
            chunk_text = f"{title}: {segment}" if title else segment
            chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=chunk_text))
            chunk_id += 1

            # Create sliding window overlap using whole sentences
            overlap_chunk = []
            overlap_words = 0
            for s in reversed(current_chunk):
                s_len = len(s.split())
                if overlap_words + s_len <= target_overlap_words or not overlap_chunk:
                    overlap_chunk.insert(0, s)
                    overlap_words += s_len
                else:
                    break

            current_chunk = overlap_chunk
            current_words = overlap_words

        current_chunk.append(sentence)
        current_words += sentence_words

    # Catch the final chunk
    if current_chunk:
        segment = " ".join(current_chunk)
        chunk_text = f"{title}: {segment}" if title else segment
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=chunk_text))

    return chunks