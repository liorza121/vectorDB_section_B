"""Optimized Retrieval pipeline with Broad Recall and Top-K Summation."""
from __future__ import annotations

import heapq
from collections import defaultdict
from pathlib import Path
from typing import List, Optional
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
        _CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', max_length=256)
    return _CROSS_ENCODER


def _get_index(artifacts_dir=None):
    global _FAISS_INDEX, _PAGE_IDS, _META_DICT
    if _FAISS_INDEX is None:
        root = Path("artifacts") if artifacts_dir is None else Path(artifacts_dir)
        meta_path = root / INDEX_META_NAME
        _META_DICT = json.loads(meta_path.read_text(encoding="utf-8"))
        _PAGE_IDS = [int(x) for x in _META_DICT["page_ids"]]

        vectors = np.load(root / INDEX_VECTORS_NAME)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        faiss.normalize_L2(vectors)

        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        _FAISS_INDEX = index

    return _FAISS_INDEX, _PAGE_IDS, _META_DICT


def search_batch(
        queries: List[str],
        *,
        top_k: int = K_EVAL,
        rerank_pool_size: int = 60,  # Number of unique PAGES to rerank, not chunks
        artifacts_dir: Optional[Path] = None,
        **kwargs  # Absorb any loose kwargs from main.py like 'pool='
) -> List[List[int]]:

    # Check if main.py is passing 'pool' instead of 'rerank_pool_size'
    if 'pool' in kwargs:
        rerank_pool_size = kwargs['pool']

    index, page_ids, meta = _get_index(artifacts_dir)
    cross_encoder = _get_cross_encoder()

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]
    if query_vectors.dtype != np.float32:
        query_vectors = query_vectors.astype(np.float32)

    faiss.normalize_L2(query_vectors)

    # WIDEN THE NET: Retrieve a massive number of chunks from FAISS (2000)
    # This solves the "Recall Bottleneck" where synthetic templates push
    # the ground truth document completely out of the candidate pool.
    faiss_k = min(2000, index.ntotal)
    faiss_dists, faiss_indices = index.search(query_vectors, k=faiss_k)

    ranked: List[List[int]] = []
    all_cross_inputs = []
    query_boundaries = []

    for i, (query, row_indices, row_dists) in enumerate(zip(queries, faiss_indices, faiss_dists)):
        page_to_chunks = defaultdict(list)
        page_max_faiss = defaultdict(float)

        # Group the 2000 retrieved chunks by their parent Page ID
        for idx, dist in zip(row_indices, row_dists):
            if idx == -1:
                continue
            pid = page_ids[idx]
            page_to_chunks[pid].append(idx)

            # Keep the maximum FAISS score for each page
            if dist > page_max_faiss[pid]:
                page_max_faiss[pid] = float(dist)

        # Select the Top `rerank_pool_size` PAGES based on their best FAISS chunk
        top_candidate_pages = heapq.nlargest(
            rerank_pool_size,
            page_max_faiss.keys(),
            key=lambda p: page_max_faiss[p]
        )

        query_chunks_count = 0
        query_pids = []

        # For each candidate page, extract up to 3 chunks to send to the Cross-Encoder
        for pid in top_candidate_pages:
            # The lists in page_to_chunks are implicitly sorted by FAISS rank
            selected_chunk_idxs = page_to_chunks[pid][:3]

            for idx in selected_chunk_idxs:
                text = meta["chunk_texts"][idx]
                all_cross_inputs.append([query, text])
                query_pids.append(pid)
                query_chunks_count += 1

        query_boundaries.append((query_chunks_count, query_pids))

    # Bulk Cross-Encoder Reranking
    all_ce_scores = cross_encoder.predict(all_cross_inputs, batch_size=128)

    # Unpack and Aggregate with Top-K Summation
    current_idx = 0
    for num_candidates, candidate_page_ids in query_boundaries:
        if num_candidates == 0:
            ranked.append([])
            continue

        ce_scores = all_ce_scores[current_idx : current_idx + num_candidates]
        current_idx += num_candidates

        # Group Cross-Encoder scores by page ID
        page_ce_scores = defaultdict(list)
        for score, pid in zip(ce_scores, candidate_page_ids):
            page_ce_scores[pid].append(float(score))

        page_final_scores = {}
        for pid, scores in page_ce_scores.items():
            # Sum the top 3 Cross-Encoder chunk scores for the page
            scores.sort(reverse=True)
            page_final_scores[pid] = sum(scores[:3])

        # Select final Top 10 pages for the query
        top_pages = heapq.nlargest(
            top_k,
            page_final_scores.keys(),
            key=lambda p: page_final_scores[p]
        )
        ranked.append(top_pages)

    return ranked