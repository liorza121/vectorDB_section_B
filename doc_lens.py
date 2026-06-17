import json
import os
from pathlib import Path
from collections import Counter
import multiprocessing

# Adjust this if your directory structure is different
WIKI_DATA_DIR = Path("data/Wikipedia Entries")


def count_words(file_path: Path) -> int:
    """Reads a JSON file and returns the word count of its 'content'."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = data.get("content", "")
            # Simple whitespace split is sufficient for a length estimate
            return len(text.split())
    except Exception:
        return 0


def generate_histogram(sample_size: int = None):
    print(f"Scanning directory: {WIKI_DATA_DIR}...")
    files = list(WIKI_DATA_DIR.glob("*.json"))

    if not files:
        print("Error: No JSON files found. Check your WIKI_DATA_DIR path.")
        return

    if sample_size and sample_size < len(files):
        import random
        files_to_scan = random.sample(files, sample_size)
        print(f"Analyzing random sample of {sample_size} files out of {len(files)}...")
    else:
        files_to_scan = files
        print(f"Analyzing all {len(files)} files...")

    # Use multiprocessing to speed up the read process
    with multiprocessing.Pool() as pool:
        word_counts = pool.map(count_words, files_to_scan)

    # Filter out empty files or read errors
    valid_counts = [c for c in word_counts if c > 0]

    if not valid_counts:
        print("No valid content found to analyze.")
        return

    # Calculate basic statistics
    valid_counts.sort()
    n = len(valid_counts)
    min_val = valid_counts[0]
    max_val = valid_counts[-1]
    avg_val = sum(valid_counts) / n
    p50 = valid_counts[int(n * 0.50)]
    p75 = valid_counts[int(n * 0.75)]
    p90 = valid_counts[int(n * 0.90)]
    p99 = valid_counts[int(n * 0.99)]

    print("\n--- Document Word Count Statistics ---")
    print(f"Total Documents Analyzed: {n}")
    print(f"Minimum Length: {min_val} words")
    print(f"Average Length: {avg_val:.1f} words")
    print(f"Median (50th %): {p50} words")
    print(f"75th Percentile: {p75} words")
    print(f"90th Percentile: {p90} words")
    print(f"99th Percentile: {p99} words")
    print(f"Maximum Length: {max_val} words")

    # Generate a simple text-based histogram
    print("\n--- Word Count Histogram ---")

    # Define bucket boundaries (adjust these based on the statistics if needed)
    buckets = [
        (0, 50), (51, 100), (101, 150), (151, 200),
        (201, 250), (251, 300), (301, 400), (401, 500),
        (501, 1000), (1001, float('inf'))
    ]

    bucket_counts = {b: 0 for b in buckets}

    for count in valid_counts:
        for b in buckets:
            if b[0] <= count <= b[1]:
                bucket_counts[b] += 1
                break

    # Find the max bucket size to scale the visual bars
    max_bucket_size = max(bucket_counts.values())
    if max_bucket_size == 0: max_bucket_size = 1  # Prevent division by zero

    max_bar_length = 50

    for b in buckets:
        count = bucket_counts[b]
        label = f"{b[0]}-{b[1] if b[1] != float('inf') else '+'}"

        # Calculate bar length
        bar_length = int((count / max_bucket_size) * max_bar_length)
        bar = "█" * bar_length

        # Format the output for readability
        print(f"{label:>10} words | {count:>7} docs | {bar}")


if __name__ == "__main__":
    # You can set sample_size=10000 to run it quickly, or None to run the full corpus
    generate_histogram(sample_size=None)