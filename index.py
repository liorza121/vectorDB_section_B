"""Offline index build and load with Cross-Encoder text caching."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Note: We import chunk_entry directly now instead of chunk_corpus
from chunk import Chunk, chunk_entry
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"
BATCH_SIZE = 10000  # Number of chunks to hold in memory before embedding


def build_index(
        *,
        entries_dir: Optional[Path] = None,
        artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Builds a standard 384-dimensional single-vector dense index using a
    memory-safe batched streaming pipeline.
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()

    # Master lists to hold the final consolidated data
    all_page_ids: List[int] = []
    all_chunk_ids: List[int] = []
    all_chunk_texts: List[str] = []
    all_vector_batches: List[np.ndarray] = []

    # Temporary buckets for the current batch
    current_batch_texts: List[str] = []
    current_batch_page_ids: List[int] = []
    current_batch_chunk_ids: List[int] = []

    print("Starting batched index build...", flush=True)

    # 1. Stream entries one by one (No list() wrapper!)
    for record in iter_entries(entries_dir):
        chunks: List[Chunk] = chunk_entry(record)

        for c in chunks:
            current_batch_texts.append(c.text)
            current_batch_page_ids.append(c.page_id)
            current_batch_chunk_ids.append(c.chunk_id)

        # 2. When the bucket is full, embed it and flush the RAM
        if len(current_batch_texts) >= BATCH_SIZE:
            print(f"Embedding batch of {len(current_batch_texts)} chunks...", flush=True)
            vectors = embed_texts(current_batch_texts)

            if vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)

            all_vector_batches.append(vectors)
            all_page_ids.extend(current_batch_page_ids)
            all_chunk_ids.extend(current_batch_chunk_ids)
            all_chunk_texts.extend(current_batch_texts)

            # Clear the temporary buckets to free memory
            current_batch_texts = []
            current_batch_page_ids = []
            current_batch_chunk_ids = []

    # 3. Catch any leftover chunks in the final partial batch
    if current_batch_texts:
        print(f"Embedding final batch of {len(current_batch_texts)} chunks...", flush=True)
        vectors = embed_texts(current_batch_texts)

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        all_vector_batches.append(vectors)
        all_page_ids.extend(current_batch_page_ids)
        all_chunk_ids.extend(current_batch_chunk_ids)
        all_chunk_texts.extend(current_batch_texts)

    # 4. Consolidate all the individual vector batches into one massive matrix
    print("Concatenating vector batches...", flush=True)
    if all_vector_batches:
        final_vectors = np.vstack(all_vector_batches)
    else:
        final_vectors = np.empty((0, 384), dtype=np.float32)

    # Save the raw numpy matrix artifact
    print("Saving vectors to disk...", flush=True)
    np.save(out_dir / INDEX_VECTORS_NAME, final_vectors)

    # 5. Store mappings AND the raw text payload needed for Stage-2 re-ranking
    print("Saving metadata to disk...", flush=True)
    meta = {
        "page_ids": all_page_ids,
        "chunk_ids": all_chunk_ids,
        "chunk_texts": all_chunk_texts,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(all_page_ids),
    }

    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("Index compilation complete! Meta and vectors saved securely.", flush=True)
    return final_vectors, all_page_ids


def load_index(
        artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Load precomputed vectors and page_id map from artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / INDEX_VECTORS_NAME)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    return vectors, page_ids