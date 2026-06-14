"""Instant query-time aggregation parameter sweeper for alpha."""
from __future__ import annotations

import sys
from eval import load_query_file, evaluate_run
from retrieve import search_batch
from utils import PUBLIC_QUERIES_PATH


def run_alpha_sweep():
    # Load the 50 public validation queries and ground truth data
    print("Loading public validation queries...", flush=True)
    gt_rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in gt_rows]
    ground_truth = [r["relevant_page_ids"] for r in gt_rows]

    # Test values from 0.0 (pure MaxP) up to 0.5 (heavy multi-chunk weighting)
    alpha_candidates = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    best_ndcg = -1.0
    best_alpha = 0.0
    results = []

    print("\nStarting Instant Aggregation Sweep...")
    print(f"{'Alpha Value':<15} | {'Public NDCG@10':<15}")
    print("-" * 35)

    for alpha in alpha_candidates:
        # Wrap search_batch using a lambda to inject the current alpha value dynamically
        run_fn = lambda q: search_batch(q, alpha=alpha)

        # Evaluate current run metrics
        metrics = evaluate_run(queries, ground_truth, run_fn)
        current_ndcg = metrics["mean_ndcg@10"]

        print(f"{alpha:<15} | {current_ndcg:.4f}", flush=True)
        results.append((alpha, current_ndcg))

        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_alpha = alpha

    print("-" * 35)
    print(f"🏆 Best Configuration: alpha = {best_alpha} (NDCG = {best_ndcg:.4f})")
    print("\nNext Step: Set the default value of alpha in retrieve.py to this best value!", flush=True)


if __name__ == "__main__":
    run_alpha_sweep()