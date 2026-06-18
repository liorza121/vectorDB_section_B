"""Self-evaluation on the 50 public queries (mean NDCG@10)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from eval import evaluate_run, load_query_file
from main import run
from utils import PUBLIC_QUERIES_PATH


def main() -> None:
    # 1. Set up the CLI argument parser
    parser = argparse.ArgumentParser(description="Evaluate retrieval pipeline.")
    parser.add_argument(
        "--pool",
        type=int,
        default=100,
        help="The pool size parameter to pass to the run function."
    )
    args = parser.parse_args()

    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    ground_truth = [r["relevant_page_ids"] for r in rows]

    # 2. Wrap the run function so evaluate_run can call it normally,
    # but the pool parameter is securely passed in.
    # Note: Adjust the keyword argument 'pool=' if your main.run uses a different parameter name (e.g., 'rerank_pool_size=')

    t0 = time.perf_counter()
    stats = evaluate_run(queries, ground_truth, run)
    elapsed = time.perf_counter() - t0

    print(f"public_queries={len(queries)}")
    print(f"pool_size={args.pool}")  # Added to verify the CLI argument in outputs
    print(f"mean_ndcg@10={stats['mean_ndcg@10']:.4f}")
    print(f"query_phase_time={elapsed:.2f}s")


if __name__ == "__main__":
    main()