import json
from pathlib import Path
from main import run
from eval import load_query_file
from utils import PUBLIC_QUERIES_PATH

# Directory containing the original Wikipedia JSON files
WIKI_DATA_DIR = Path("data/Wikipedia Entries")

def get_doc_content(pid: int) -> str:
    """Helper to load full page content from the source JSON files."""
    doc_path = WIKI_DATA_DIR / f"{pid}.json"
    if doc_path.exists():
        with open(doc_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return f"Title: {data.get('title')}\n\nContent: {data.get('content')}"
    return f"Error: Could not find source file for page_id {pid}"

def generate_diagnostic_data(output_dir: str = "diagnostic_results"):
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Generating diagnostic report in: {output_dir}/")

    for i, row in enumerate(rows):
        query = row["query"]
        query_id = f"q_{i:03d}"
        relevant_pages = row["relevant_page_ids"]

        # Perform retrieval
        retrieved_page_ids = run([query], pool=150)[0][:10]

        query_dir = output_path / query_id
        query_dir.mkdir(exist_ok=True)
        (query_dir / "query.txt").write_text(query, encoding="utf-8")

        # 4. Create sub-directories
        (query_dir / "retrieved_docs").mkdir(exist_ok=True)
        (query_dir / "actual_labels").mkdir(exist_ok=True)

        # 5. Populate retrieved docs with full content
        for rank, pid in enumerate(retrieved_page_ids):
            content = get_doc_content(pid)
            doc_file = query_dir / "retrieved_docs" / f"rank_{rank+1}_id_{pid}.txt"
            doc_file.write_text(f"Page ID: {pid}\nRank: {rank+1}\n\n{content}", encoding="utf-8")

        # 6. Populate actual labels with full content
        for pid in relevant_pages:
            content = get_doc_content(pid)
            label_file = query_dir / "actual_labels" / f"relevant_id_{pid}.txt"
            label_file.write_text(f"Page ID: {pid}\nStatus: Ground Truth\n\n{content}", encoding="utf-8")

    print("Diagnostic generation complete.")

if __name__ == "__main__":
    generate_diagnostic_data()