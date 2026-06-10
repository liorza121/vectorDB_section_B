"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

import heapq
from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Return ranked page_id lists (best first) for each query.

    Uses the MaxP strategy: the score of a page_id is the score of its
    highest-scoring chunk. Uses a min-heap to extract the top_k elements
    efficiently without sorting the entire corpus.
    """
    corpus_vectors, page_ids = load_index(artifacts_dir)
    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    # Compute dot product similarity scores for all query-chunk pairs
    # Shape: (num_queries, num_chunks)
    scores = query_vectors @ corpus_vectors.T

    ranked: List[List[int]] = []

    for row in scores:
        # Step 1: MaxP Aggregation
        # Map each unique page_id to its maximum chunk score
        page_scores: dict[int, float] = {}
        for idx, score in enumerate(row):
            pid = int(page_ids[idx])
            if pid not in page_scores or score > page_scores[pid]:
                page_scores[pid] = score

        # Step 2: Extract the fixed number of top_k maxima using a heap
        # This scales at O(U log K) instead of O(U log U) full-sorting
        top_pages = heapq.nlargest(
            top_k,
            page_scores.keys(),
            key=lambda pid: page_scores[pid]
        )

        ranked.append(top_pages)

    return ranked