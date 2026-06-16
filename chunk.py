"""Preprocessing and chunking utilizing compliant native text splitters."""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text

@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def get_sentences(text: str) -> List[str]:
    """Pragmatic Sentence Boundary Disambiguation without external libraries."""
    # 1. Normalize whitespace to fix the paragraph/newline trap
    clean_text = re.sub(r'\s+', ' ', text).strip()

    # 2. The Expanded Protection Dictionary for Wikipedia Corpus
    protected_terms = {
        "U.S.": "U<DOT>S<DOT>", "St.": "St<DOT>", "Dr.": "Dr<DOT>",
        "v.": "v<DOT>", "vs.": "vs<DOT>", "e.g.": "e<DOT>g<DOT>",
        "i.e.": "i<DOT>e<DOT>", "Mr.": "Mr<DOT>", "Mrs.": "Mrs<DOT>",
        "Ms.": "Ms<DOT>", "Prof.": "Prof<DOT>", "Rev.": "Rev<DOT>",
        "Gen.": "Gen<DOT>", "Col.": "Col<DOT>", "Capt.": "Capt<DOT>",
        "Lt.": "Lt<DOT>", "Gov.": "Gov<DOT>", "Sen.": "Sen<DOT>",
        "Rep.": "Rep<DOT>", "Mt.": "Mt<DOT>", "Ft.": "Ft<DOT>",
        "Ph.D.": "Ph<DOT>D<DOT>", "Vol.": "Vol<DOT>", "Ed.": "Ed<DOT>",
        "No.": "No<DOT>", "B.A.": "B<DOT>A<DOT>", "M.A.": "M<DOT>A<DOT>"
    }

    # Protect single-letter initials (A. through Z.)
    for letter in string.ascii_uppercase:
        protected_terms[f" {letter}."] = f" {letter}<DOT>"

    # 3. Apply protections
    for term, protected in protected_terms.items():
        clean_text = clean_text.replace(term, protected)

    # 4. Split on Punctuation + Space + Capital/Number
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', clean_text)

    # 5. Restore the protected abbreviations
    sentences = []
    for sentence in raw_sentences:
        for term, protected in protected_terms.items():
            sentence = sentence.replace(protected, term)
        sentences.append(sentence)

    return sentences


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Split one corpus entry using a sentence-aware sliding window.
    Prevents semantic shearing by ensuring chunks always end on punctuation.
    """
    page_id = int(record["page_id"])
    title = record.get("title", "").strip()
    text = entry_text(record)

    if not text.strip():
        full_text = f"{title}:" if title else ""
        return [Chunk(page_id=page_id, chunk_id=0, text=full_text)]

    # Fetch safely extracted sentences
    sentences = get_sentences(text)

    # Hyperparameters
    target_chunk_words = 175
    target_overlap_words = 40

    chunks: List[Chunk] = []
    chunk_id = 0

    current_chunk_sentences = []
    current_word_count = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if not sentence:
            i += 1
            continue

        sentence_words = len(sentence.split())

        # If adding the next sentence keeps us under the limit, or the chunk is currently empty
        if current_word_count + sentence_words <= target_chunk_words or not current_chunk_sentences:
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_words
            i += 1
        else:
            # The chunk is full. Save it.
            segment = " ".join(current_chunk_sentences)
            chunk_text = f"{title}: {segment}" if title else segment
            chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=chunk_text))
            chunk_id += 1

            # Slide the window by dropping sentences from the front
            # until we hit our target overlap threshold.
            overlap_word_count = current_word_count
            while len(current_chunk_sentences) > 1 and overlap_word_count > target_overlap_words:
                dropped_sentence = current_chunk_sentences.pop(0)
                overlap_word_count -= len(dropped_sentence.split())

            current_word_count = overlap_word_count

    # Catch the final chunk
    if current_chunk_sentences:
        segment = " ".join(current_chunk_sentences)
        chunk_text = f"{title}: {segment}" if title else segment
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=chunk_text))

    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    """Processes the entire corpus into chunks."""
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks