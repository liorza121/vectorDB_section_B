# Section B — High-Performance Retrieval Pipeline

An end-to-end search and retrieval architecture optimized for high speed and maximum recall under strict runtime constraints.

---

### Video Demonstration Link:
https://drive.google.com/file/d/1X2rAjN77Bqm5Hb2O6DCLeH1YBoVZmmpg/view?usp=sharing
### Slides link:
https://docs.google.com/presentation/d/1vPEx3chFruAgN0z9cRehPiIbcfyITrcS-F2oS5VPTgk/edit?usp=drive_link


## 🛠️ System Architecture

Our pipeline processes raw text documents and executes targeted factual query retrieval through a four-stage process:

1. **Stage 1: Sentence Chunking**
   * **Sentence-Level Splitting:** Tokenizes text by terminal punctuation while bypassing abbreviations to preserve full grammatical unity.
   * **250-Word Target Blocks:** Aggregates isolated sentences into discrete text segments without breaking mid-sentence.
   * **80-Word Sentence Overlap:** Traverses backward through full sentences to establish smooth semantic continuity between chunks.
   * **Title Metadata Injection:** Prepends the parent document title to every chunk text to anchor global context.

2. **Stage 2: Dense Embedding**
   * **MiniLM-L6-v2 Encoder:** Translates text chunks into dense 384-dimensional vector spaces using the mandatory sentence-transformer core (`all-MiniLM-L6-v2`).
   * **64-Chunk Mini-Batches:** Feeds segments into the model in uniform groups of 64 to optimize matrix operations and inference throughput.
   * **Strict $L_2$ Normalization:** Enforces native unit-length normalization ($||v||_2 = 1$) during encoding to guarantee dot-product directional accuracy.
   * **Continuous Array Cast:** Casts all inferred batch layers into continuous 32-bit floating-point arrays to match FAISS hardware specifications.

3. **Stage 3: Offline Indexing**
   * **Batched Pipeline:** Streams JSON documents and processes them in batches of 10,000 to keep the system memory usage static and predictable.
   * **FAISS FlatIP:** Utilizes inner-product (FlatIP) architecture to execute fast cosine similarity calculations over mapped coordinate spaces.
   * **Calendar Year Indexing:** Uses Regex to catch the existence of calendar years in the document and tracks the years in every document, for later use in retrieval to match year-related queries.
   * **Rapid Online Loading:** Decouples index build steps from search steps, reducing retrieval initialization to under two seconds.

4. **Stage 4: Two-Stage Retrieval**
   * **Widen-the-Net Recall ($k=2000$):** Retrieves an expansive candidate pool of 2,000 chunks via FAISS to break through the initial recall bottleneck.
   * **Page-ID Grouping & Filtering:** Groups chunks by original page ID, and keeps chunks only from the top pages found in aggregate to avoid spurious relevance predictions.
   * **Bulk Cross-Encoder Reranking:** Feeds up to 3 candidate text chunks per page into the `ms-marco-TinyBERT-L-2-v2` encoder to generate highly precise query-text alignment scores.
   * **Top-K Summation Aggregation:** Combines the top 3 Cross-Encoder chunk scores per unique page to accurately rank and extract the final Top 10 most relevant pages.

---

## ⚙️ Setup & Installation

Ensure the raw corpus lives at **`data/Wikipedia Entries/`** (included in the handout). 

Activate your virtual environment and install the required dependencies:
```bash
cd path/to/student
source .venv/bin/activate  # If using a virtual environment
pip install -r requirements.txt
```

---

## 📦 Generated Artifacts

Running the offline building script populates the `artifacts/` directory with memory-mapped array structures and metadata. These files decouple index compilation from online querying, reducing retrieval initialization times to under two seconds:

* `artifacts/index_vectors.npy` – High-fidelity matrix containing continuous 32-bit floating-point dense embedding arrays matching FAISS hardware specifications.
* `artifacts/index_meta.json` – Structural tracking files pairing internal database IDs, extracted year tokens, and chunk texts back to their parent document source IDs.

---

## 🚀 How to Run & Verify

### 1. Build Index (Offline, not timed)
Run once locally to create the `artifacts/` directory and populate it. **You must submit these files in your repo**; the staff do not rebuild the index at grading time.
```bash
python scripts/build_index.py
```

### 2. Public Self-Test (Evaluation)
After building, verify that a fresh run successfully loads your submitted artifacts without triggering a rebuild:
```bash
python scripts/eval_public.py
```



