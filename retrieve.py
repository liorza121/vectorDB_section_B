"""Optimized Retrieval pipeline with Pre-Computed Time Index Filtering.
Refactored for modularity, clean helper functions, and zero magic numbers.
"""
from __future__ import annotations

import heapq
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from sentence_transformers import CrossEncoder

from embed import embed_queries

# =============================================================================
# Configuration Constants (No Magic Numbers)
# =============================================================================
INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"

K_EVAL = 10
DEFAULT_RERANK_POOL_SIZE = 1500
MAX_FAISS_RETRIEVAL = 2000
MAX_CHUNKS_PER_PAGE = 3

CE_MODEL_NAME = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
CE_MAX_LENGTH = 256
CE_BATCH_SIZE = 128

# =============================================================================
# Global State Cache
# =============================================================================
_FAISS_INDEX: Optional[faiss.IndexIDMap] = None
_META_DICT: Optional[dict] = None
_CHUNK_TO_PAGE: Optional[dict] = None
_CHUNK_TO_TEXT: Optional[dict] = None
_CROSS_ENCODER: Optional[CrossEncoder] = None


# =============================================================================
# Initialization Helpers
# =============================================================================
def _get_cross_encoder() -> CrossEncoder:
    """Lazy load the Cross-Encoder model."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = CrossEncoder(CE_MODEL_NAME, max_length=CE_MAX_LENGTH)
    return _CROSS_ENCODER


def _get_index(artifacts_dir: Optional[Path] = None) -> Tuple[faiss.IndexIDMap, dict, dict, dict]:
    """Lazy load the FAISS index and metadata mappings."""
    global _FAISS_INDEX, _META_DICT, _CHUNK_TO_PAGE, _CHUNK_TO_TEXT

    if _FAISS_INDEX is None:
        root = Path("artifacts") if artifacts_dir is None else Path(artifacts_dir)
        meta_path = root / INDEX_META_NAME
        _META_DICT = json.loads(meta_path.read_text(encoding="utf-8"))

        num_chunks = len(_META_DICT["chunk_texts"])
        global_ids = list(range(num_chunks))

        _CHUNK_TO_PAGE = {gid: pid for gid, pid in enumerate(_META_DICT["page_ids"])}
        _CHUNK_TO_TEXT = {gid: text for gid, text in enumerate(_META_DICT["chunk_texts"])}

        vectors = np.load(root / INDEX_VECTORS_NAME)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        faiss.normalize_L2(vectors)
        dimension = vectors.shape[1]

        base_index = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIDMap(base_index)

        chunk_ids_array = np.array(global_ids, dtype=np.int64)
        index.add_with_ids(vectors, chunk_ids_array)

        _FAISS_INDEX = index

    return _FAISS_INDEX, _META_DICT, _CHUNK_TO_PAGE, _CHUNK_TO_TEXT


# =============================================================================
# Query Processing & Filtering Helpers
# =============================================================================
def extract_time_constraints(query: str) -> List[Set[str]]:
    """Extracts exact years and expands decades into valid year sets."""
    constraints = []

    # 1. Match exact 4-digit years NOT followed by an 's' (e.g., "1987")
    year_matches = re.finditer(r'\b((?:17|18|19|20)\d{2})\b(?!\s*s)', query)
    for match in year_matches:
        constraints.append({match.group(1)})

    # 2. Match decades (e.g., "1820s")
    decade_matches = re.finditer(r'\b((?:17|18|19|20)\d)0s\b', query)
    for match in decade_matches:
        base_year = int(match.group(1) + "0")
        valid_terms = {f"{base_year}s"} | {str(y) for y in range(base_year, base_year + 10)}
        constraints.append(valid_terms)

    return constraints


def _build_search_parameters(
    query: str, time_index: dict
) -> Tuple[Optional[faiss.SearchParameters], Optional[np.ndarray], Optional[faiss.IDSelectorArray]]:
    """
    Builds a FAISS IDSelectorArray based on extracted time constraints.
    CRITICAL: Returns the numpy array and selector object to prevent Python's GC
    from destroying them before FAISS executes the search in C++.
    """
    constraints = extract_time_constraints(query)
    if not constraints:
        return None, None, None

    valid_ids_set = None

    for constraint_group in constraints:
        group_ids = set()
        for term in constraint_group:
            if term in time_index:
                group_ids.update(time_index[term])

        if valid_ids_set is None:
            valid_ids_set = group_ids
        else:
            valid_ids_set.intersection_update(group_ids)

    if valid_ids_set:
        valid_ids_array = np.array(list(valid_ids_set), dtype=np.int64)
    else:
        valid_ids_array = np.array([], dtype=np.int64)

    sel = faiss.IDSelectorArray(valid_ids_array.size, faiss.swig_ptr(valid_ids_array))
    params = faiss.SearchParameters(sel=sel)

    return params, valid_ids_array, sel


# =============================================================================
# Aggregation & Scoring Helpers
# =============================================================================
def _filter_top_pages_from_faiss(
    row_chunk_ids: np.ndarray,
    row_dists: np.ndarray,
    chunk_to_page: dict,
    pool_size: int
) -> Tuple[List[int], Dict[int, List[int]]]:
    """Groups FAISS chunk results by Page ID and isolates the top candidate pages."""
    page_to_chunks = defaultdict(list)
    page_max_faiss = defaultdict(float)

    for cid, dist in zip(row_chunk_ids, row_dists):
        if cid == -1:
            continue
        pid = chunk_to_page[cid]
        page_to_chunks[pid].append(cid)

        if dist > page_max_faiss[pid]:
            page_max_faiss[pid] = float(dist)

    top_candidate_pages = heapq.nlargest(
        pool_size,
        page_max_faiss.keys(),
        key=lambda p: page_max_faiss[p]
    )

    return top_candidate_pages, dict(page_to_chunks)


def _aggregate_cross_encoder_scores(
    ce_scores: np.ndarray,
    candidate_page_ids: List[int],
    top_k: int
) -> List[int]:
    """Sums top chunk scores per page to determine final page relevance."""
    page_ce_scores = defaultdict(list)
    for score, pid in zip(ce_scores, candidate_page_ids):
        page_ce_scores[pid].append(float(score))

    page_final_scores = {}
    for pid, scores in page_ce_scores.items():
        scores.sort(reverse=True)
        page_final_scores[pid] = sum(scores[:MAX_CHUNKS_PER_PAGE])

    top_pages = heapq.nlargest(
        top_k,
        page_final_scores.keys(),
        key=lambda p: page_final_scores[p]
    )
    return top_pages


# =============================================================================
# Main Pipeline
# =============================================================================
def search_batch(
        queries: List[str],
        *,
        top_k: int = K_EVAL,
        artifacts_dir: Optional[Path] = None,
        **kwargs
) -> List[List[int]]:
    """Executes the two-stage logic/semantic retrieval pipeline."""

    rerank_pool_size = kwargs.get('pool', DEFAULT_RERANK_POOL_SIZE)

    index, meta, chunk_to_page, chunk_to_text = _get_index(artifacts_dir)
    cross_encoder = _get_cross_encoder()

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]
    if query_vectors.dtype != np.float32:
        query_vectors = query_vectors.astype(np.float32)

    faiss.normalize_L2(query_vectors)
    faiss_k = min(MAX_FAISS_RETRIEVAL, index.ntotal)
    time_index = meta.get("time_index", {})

    all_cross_inputs = []
    query_boundaries = []

    # ---------------------------------------------------------
    # Stage 1: Pre-filtered FAISS Retrieval
    # ---------------------------------------------------------
    for i, query in enumerate(queries):

        # The variables _keepalive_array and _keepalive_sel are explicitly scoped
        # here so they are not destroyed before index.search() executes.
        params, _keepalive_array, _keepalive_sel = _build_search_parameters(query, time_index)
        query_vector = query_vectors[i].reshape(1, -1)

        if params:
            faiss_dists, faiss_indices = index.search(
                query_vector,
                k=faiss_k,
                params=params
            )
        else:
            faiss_dists, faiss_indices = index.search(
                query_vector,
                k=faiss_k
            )

        top_candidate_pages, page_to_chunks = _filter_top_pages_from_faiss(
            faiss_indices[0],
            faiss_dists[0],
            chunk_to_page,
            rerank_pool_size
        )

        query_chunks_count = 0
        query_pids = []

        for pid in top_candidate_pages:
            selected_chunk_ids = page_to_chunks[pid][:MAX_CHUNKS_PER_PAGE]

            for cid in selected_chunk_ids:
                all_cross_inputs.append([query, chunk_to_text[cid]])
                query_pids.append(pid)
                query_chunks_count += 1

        query_boundaries.append((query_chunks_count, query_pids))

    if not all_cross_inputs:
        return [[] for _ in queries]

    # ---------------------------------------------------------
    # Stage 2: Cross-Encoder Reranking
    # ---------------------------------------------------------
    all_ce_scores = cross_encoder.predict(all_cross_inputs, batch_size=CE_BATCH_SIZE)

    ranked: List[List[int]] = []
    current_idx = 0

    for num_candidates, candidate_page_ids in query_boundaries:
        if num_candidates == 0:
            ranked.append([])
            continue

        ce_scores = all_ce_scores[current_idx : current_idx + num_candidates]
        current_idx += num_candidates

        top_pages = _aggregate_cross_encoder_scores(ce_scores, candidate_page_ids, top_k)
        ranked.append(top_pages)

    return ranked