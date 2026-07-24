[README (2).md](https://github.com/user-attachments/files/30338032/README.2.md)
# 💰 Personal Finance Intelligence — RAG Chatbot

A local-first personal finance pipeline that ingests bank/UPI statements (PDF/CSV) or manual entries, runs them through a **Medallion architecture** (Bronze → Silver → Gold), auto-categorizes transactions with an LLM, and exposes a **Streamlit dashboard + RAG chatbot** for natural-language queries over your own spending data.

Everything runs locally. No data leaves your machine except the transaction descriptions sent to Groq for categorization and chat.

---

## Architecture

```
Upload (PDF/CSV) ──┐
                    ├──► Bronze (raw extraction) ──► Silver (validate + dedupe) ──┐
Manual entry ───────┘                                                             ├──► Gold (LLM categorization)
                                                                                    │
                                                              ┌─────────────────────┴─────────────────────┐
                                                              ▼                                             ▼
                                                     Dashboard (Streamlit)                     Vector Index (ChromaDB)
                                                                                                             │
                                                                                                             ▼
                                                                                              RAG Chatbot (Groq LLM)
```

| Layer | Responsibility | Engine |
|---|---|---|
| **Bronze** | Extract raw rows from PDF/CSV, enforce schema, dedupe by content hash | PySpark |
| **Silver** | Parse dates/amounts, clean text, business-key dedupe, quarantine bad rows | PySpark |
| **Gold** | LLM-categorize transactions (cached by hash), merge in manual entries | pandas + Groq |
| **Vector Index** | Embed transactions, semantic search | ChromaDB + sentence-transformers |
| **Chatbot** | Natural-language Q&A over your transactions | Groq (`openai/gpt-oss-120b`) |
| **UI** | Upload, manual entry, pipeline controls, dashboard, chat | Streamlit |

---

## Project Structure

```
finance_rag_chatbot/
├── app.py                          # Streamlit entrypoint
├── requirements.txt
├── .env                            # GROQ_API_KEY (not committed)
├── .env.example
├── .streamlit/
│   └── config.toml                 # theme + upload size limit
├── scripts/
│   ├── run_bronze_pipeline.py      # CLI: Bronze ETL (+ optional synthetic data gen)
│   ├── run_silver_pipeline.py      # CLI: Silver transform
│   └── run_gold_pipeline.py        # CLI: Gold categorization
├── test_rag_pipeline.py            # CLI: build/refresh vector index + test chat query
├── src/
│   ├── config.py                   # all paths & constants — single source of truth
│   ├── bronze/
│   │   ├── pdf_extractor.py
│   │   ├── csv_extractor.py
│   │   ├── pipeline.py
│   │   ├── spark_session.py
│   │   ├── writer.py
│   │   └── upload_handler.py       # persists UI uploads to data/uploads/
│   ├── silver/
│   │   ├── transform.py
│   │   ├── pipeline.py
│   │   └── writer.py
│   ├── gold/
│   │   ├── categorizer.py          # Groq batch categorization + retry logic
│   │   ├── pipeline.py
│   │   ├── writer.py
│   │   └── manual_overrides.py     # manual transaction store + category overrides
│   ├── rag/
│   │   ├── embeddings.py           # sentence-transformers wrapper
│   │   ├── vector_store.py         # ChromaDB persistent client
│   │   ├── indexer.py              # syncs vector index from Gold
│   │   └── chat.py                 # RAG query → Groq answer
│   └── synthetic_data/
│       └── generate_statements.py  # synthetic PDF/CSV generator for testing
└── data/                           # created automatically on first run
    ├── uploads/
    ├── synthetic/
    ├── bronze/
    ├── silver/
    ├── gold/
    ├── manual/
    └── vector_store/
```

---

## Prerequisites

| Requirement | Why | Check |
|---|---|---|
| **Python 3.10 or 3.11** | Project tested on these; avoid the Microsoft Store build of Python (see Troubleshooting) | `python --version` |
| **JDK 11 or 17** | PySpark (Bronze/Silver layers) requires a real JVM | `java -version` |
| **Groq API key (free)** | LLM categorization + chat | [console.groq.com/keys](https://console.groq.com/keys) |
| **~2 GB free disk** | torch + sentence-transformers + model weights | — |

**Windows-specific:** install Python from [python.org](https://python.org), **not** the Microsoft Store — the Store build sandboxes file I/O in a way that silently breaks pandas/PyArrow writes to real paths, especially inside OneDrive-synced folders.

---

## Setup — from a clean clone

```powershell
# 1. Clone and enter the project
git clone <your-repo-url> finance_rag_chatbot
cd finance_rag_chatbot

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure your Groq API key
copy .env.example .env         # Windows: copy | macOS/Linux: cp
# then edit .env and set:
#   GROQ_API_KEY=your_key_here

# 5. Install a JDK if you don't have one (Windows)
#    Download Eclipse Temurin JDK 17: https://adoptium.net/
#    Set JAVA_HOME to the install folder, add %JAVA_HOME%\bin to PATH
java -version                  # confirm it resolves
```

---

## Running the project — first-time walkthrough

Run these **in order**. Each stage depends on the one before it.

### Step 1 — Generate test data (optional, skip if you have real statements)

```powershell
python scripts/run_bronze_pipeline.py --generate 3
```
Generates 3 synthetic PDF statements + 1 synthetic UPI-style CSV into `data/synthetic/`, then ingests them into Bronze in the same run.

### Step 2 — Bronze (if you skipped Step 1, or to ingest real files)

```powershell
python scripts/run_bronze_pipeline.py                  # ingest data/synthetic/
python scripts/run_bronze_pipeline.py --source uploads  # ingest data/uploads/ instead
```
Expect: `✅ Bronze batch written: data/bronze/bronze_<batch_id>.parquet`

### Step 3 — Silver

```powershell
python scripts/run_silver_pipeline.py
```
Expect: `✅ Silver transactions written: ...` and either `✅ No quarantined rows` or a note about rows needing review.

### Step 4 — Gold (LLM categorization)

```powershell
python scripts/run_gold_pipeline.py
```
Expect: batches logged with progress, then `✅ Gold transactions written: data/gold/gold_transactions.parquet`. This is the step that calls Groq — first run will take a few minutes for a large batch (rate-limit pacing is intentional, see `BATCH_PAUSE_SECONDS` in `src/gold/pipeline.py`). Reruns are near-instant — categorization results are cached by transaction hash.

### Step 5 — Build the vector index

```powershell
python test_rag_pipeline.py
```
Expect: `{'upserted': N, 'removed_stale': 0}` followed by a test chat answer citing real numbers from your data.

> **First run on Windows may throw `WinError 1114` (DLL init failure).** This is a one-time torch/pandas DLL load-order race. Just rerun the same command — see Troubleshooting below.

### Step 6 — Launch the app

```powershell
streamlit run app.py
```
Opens at `http://localhost:8501`. From here you can:
- Upload new statements or log manual transactions
- Click **Run Full Pipeline** to process new uploads (Bronze → Silver → Gold → vector index, all in one click)
- Click **Run Quick Refresh** after logging manual entries (Gold → index only, skips Spark)
- Use **🔁 Rebuild Index** on the dashboard to manually re-sync the vector index at any time
- Browse the **Dashboard** for KPIs, category breakdown, and monthly cash flow
- Ask questions in **💬 Ask Your Finances** — e.g. *"how much did I spend on food last month"*, *"show me my Uber transactions"*

---

## Day-to-day usage (after first-time setup)

You generally don't need the CLI scripts again — the UI wraps the whole pipeline:

1. `streamlit run app.py`
2. Upload a statement → **Save & Stage** → **Run Full Pipeline**, *or* log a manual transaction → **Run Quick Refresh**
3. Ask the chatbot anything about your data

---

## Configuration

| File | Purpose |
|---|---|
| `.env` | `GROQ_API_KEY` — never commit this |
| `.streamlit/config.toml` | Theme + `maxUploadSize` (default 25 MB) |
| `src/config.py` | All data paths, category list, embedding model name — single source of truth for every other module |

To change the categorization/chat model, edit `GROQ_MODEL` in `src/gold/categorizer.py` and `src/rag/chat.py`.

---

## Troubleshooting

**`WinError 1114` — DLL initialization failed (torch/c10.dll)**
A one-time import-order race between pandas' and torch's bundled native runtimes on Windows. Fix: ensure `import torch` is the very first import in any entrypoint file (`app.py`, `test_rag_pipeline.py`) — before `pandas`, `streamlit`, or anything from `src.rag`/`src.gold`. If it still fails once, just rerun the same command; the second attempt typically succeeds once the DLL is cached.

**Crash with exit code `-1073741819` during embedding**
Native thread-pool collision (torch OpenMP + another loaded runtime). Confirm these lines are the *first* thing executed in `src/rag/embeddings.py`, before any chromadb/sentence-transformers import:
```python
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
```
If it persists, do a clean reinstall of `torch`, `sentence-transformers`, and `chromadb` in that order.

**`JAVA_HOME is not set`**
Install a JDK (see Prerequisites), set `JAVA_HOME`, add `%JAVA_HOME%\bin` to `PATH`, restart your terminal.

**Directory "exists" but writes fail silently, especially inside OneDrive**
You're likely on the Microsoft Store build of Python. Reinstall from [python.org](https://python.org), recreate the venv, and ideally move the project outside any cloud-synced folder.

**429 Too Many Requests during Gold categorization**
Expected occasionally under load — `tenacity` retries with backoff automatically. If it happens on *every* batch, check `BATCH_PAUSE_SECONDS` in `src/gold/pipeline.py` against Groq's current published rate limits for your model at [console.groq.com/docs](https://console.groq.com/docs).

**Chatbot says "no relevant transactions found" but data clearly exists**
The vector index is out of sync with Gold. Click **🔁 Rebuild Index** in the dashboard, or run `python test_rag_pipeline.py`.

---

## Notes on design decisions

- **Gold is fully rebuilt every run**, but the LLM category cache (`data/gold/_category_cache.json`) means only genuinely new transactions are ever sent to Groq.
- **Manual entries never pass through Bronze/Silver** — they're structured data already; routing them through a PDF/CSV extractor would add risk for no benefit. They get their own hash (derived from a UUID, not business fields) so they can never collide with each other or with statement-derived transactions.
- **An explicit manual category always skips the LLM** — it's written straight into the cache and is never second-guessed by categorization.
- **The vector index is incrementally synced**, not rebuilt from scratch — only new/removed transactions touch the embedding model on each sync.

---

## License

Personal project — add a license here if you intend to make this public (MIT is a common default for portfolio projects).
