"""Query-time retrieval reverted to clean, single-vector baseline mapping."""
from __future__ import annotations

import heapq
from pathlib import Path
from typing import List, Optional
from collections import defaultdict
import faiss
import numpy as np
import json

from sentence_transformers import CrossEncoder

from embed import embed_queries

# Global variables for lazy loading cache
_FAISS_INDEX: Optional[faiss.IndexFlatIP] = None
_PAGE_IDS: Optional[List[int]] = None
_META_DICT: Optional[dict] = None

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"
K_EVAL = 10
_CROSS_ENCODER = None
def _get_cross_encoder() -> CrossEncoder:
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        # Swap MiniLM for TinyBERT for massive speed gains
        _CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', max_length=256)
    return _CROSS_ENCODER


def _get_index(artifacts_dir=None):
    """
    Safely loads and caches standard 384-dimensional dense vectors,
    page mappings, and metadata in global RAM.
    """
    global _FAISS_INDEX, _PAGE_IDS, _META_DICT

    if _FAISS_INDEX is None:
        root = Path("artifacts") if artifacts_dir is None else Path(artifacts_dir)

        # 1. Load the core metadata configuration
        meta_path = root / INDEX_META_NAME
        _META_DICT = json.loads(meta_path.read_text(encoding="utf-8"))
        _PAGE_IDS = [int(x) for x in _META_DICT["page_ids"]]

        # 2. Load pristine single-embedding chunk matrix (Shape: N x 384)
        vectors = np.load(root / INDEX_VECTORS_NAME)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # 3. Setup clean flat Inner Product FAISS space
        dimension = vectors.shape[1]  # Confirms 384 dimensions
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        _FAISS_INDEX = index

    return _FAISS_INDEX, _PAGE_IDS, _META_DICT


def search_batch(
        queries: List[str],
        *,
        top_k: int = K_EVAL,
        rerank_pool_size: int = 150,
        artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Executes an optimized Two-Stage Retrieval pipeline:
    FAISS Candidate Generation -> Bulk Cross-Encoder Reranking -> MaxP Aggregation.
    """
    index, page_ids, meta = _get_index(artifacts_dir)
    cross_encoder = _get_cross_encoder()
    query_vectors = embed_queries(queries)

    if query_vectors.size == 0:
        return [[] for _ in queries]

    if query_vectors.dtype != np.float32:
        query_vectors = query_vectors.astype(np.float32)

    # STAGE 1: Fast FAISS scan for a broader pool of candidates
    faiss_k = min(rerank_pool_size, index.ntotal)
    _, faiss_indices = index.search(query_vectors, k=faiss_k)

    ranked: List[List[int]] = []

    # Flattening structures for bulk inference
    all_cross_inputs = []
    query_boundaries = []

    # 1. Build a single massive list of all query-document pairs
    for i, (query, row_indices) in enumerate(zip(queries, faiss_indices)):
        valid_indices = [idx for idx in row_indices if idx != -1]

        if not valid_indices:
            query_boundaries.append((0, []))
            continue

        # Extract the raw text and page IDs for the candidate chunks
        candidate_texts = [meta["chunk_texts"][idx] for idx in valid_indices]
        candidate_page_ids = [page_ids[idx] for idx in valid_indices]

        # Format required by sentence-transformers: [[query, text1], [query, text2]]
        cross_inputs = [[query, text] for text in candidate_texts]
        all_cross_inputs.extend(cross_inputs)

        # Save how many candidates belong to this specific query
        query_boundaries.append((len(valid_indices), candidate_page_ids))

    # STAGE 2: Bulk Cross-Encoder Reranking
    # Passing the entire flattened list to predict at once maximizes hardware utilization.
    # Note: You can tweak `batch_size` up or down depending on your GPU VRAM.
    all_ce_scores = cross_encoder.predict(all_cross_inputs, batch_size=256)

    # STAGE 3: Unpack and MaxP Aggregation
    current_idx = 0
    for num_candidates, candidate_page_ids in query_boundaries:
        if num_candidates == 0:
            ranked.append([])
            continue

        # Slice the scores belonging to this specific query from the master array
        ce_scores = all_ce_scores[current_idx: current_idx + num_candidates]
        current_idx += num_candidates

        # Standard MaxP aggregation using the highly accurate CE scores
        ce_page_scores = {}
        for score, pid in zip(ce_scores, candidate_page_ids):
            if pid not in ce_page_scores or score > ce_page_scores[pid]:
                ce_page_scores[pid] = float(score)

        # Sort pages by their highest scoring chunk
        top_pages = heapq.nlargest(
            top_k,
            ce_page_scores.keys(),
            key=lambda p: ce_page_scores[p]
        )
        ranked.append(top_pages)

    return ranked