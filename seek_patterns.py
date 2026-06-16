import json
import re
from pathlib import Path
from collections import Counter
import multiprocessing

# Adjust this to your actual Wikipedia Entries directory
CORPUS_DIR = Path("data/Wikipedia Entries")


def find_anomalies(file_path: Path) -> dict:
    """Scans a single JSON file for potential regex breakers."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = data.get("content", "")

            # 1. Hunt for Abbreviations
            # Looks for 1 to 4 letter words ending in a period, followed by a space and a capital letter.
            # Example match: "U.S. " or "Dr. "
            abbrev_pattern = re.compile(r'\b([A-Za-z\.]{1,4}\.)\s+[A-Z]')
            abbreviations = abbrev_pattern.findall(text)

            # 2. Hunt for Structural Oddities (Optional, but good for sanity checks)
            # Count how often paragraphs are separated by single vs double newlines
            single_newlines = len(re.findall(r'[^\n]\n[^\n]', text))
            double_newlines = len(re.findall(r'\n\n', text))

            return {
                "abbreviations": abbreviations,
                "structural": {"single_nl": single_newlines, "double_nl": double_newlines}
            }
    except Exception as e:
        return {"abbreviations": [], "structural": {}}


def run_corpus_sweep(sample_size: int = 10000):
    """Sweeps the corpus and aggregates the most common regex breakers."""
    files = list(CORPUS_DIR.glob("*.json"))

    # Take a random sample if you don't want to wait for all 570K files
    import random
    files_to_scan = random.sample(files, min(sample_size, len(files)))

    print(f"Scanning {len(files_to_scan)} files...")

    master_abbrev_counter = Counter()

    # Use multiprocessing to chew through the files faster
    with multiprocessing.Pool() as pool:
        results = pool.map(find_anomalies, files_to_scan)

    for res in results:
        master_abbrev_counter.update(res["abbreviations"])

    print("\n--- Top 50 Potential Abbreviations to Protect ---")
    for abbrev, count in master_abbrev_counter.most_common(50):
        print(f"{abbrev} : found {count} times")


if __name__ == "__main__":
    # Start with a 10,000 file sample to see what bubbles to the top
    run_corpus_sweep(10000)